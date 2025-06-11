use image::{ImageBuffer, Rgba, ImageEncoder, ColorType};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::Read;
use clap::{Arg, Command, ArgMatches};

/// 获取推荐的线程数
fn get_recommended_threads() -> usize {
    let cpu_count = num_cpus::get();
    
    // 对于图像处理任务，考虑到I/O操作，建议线程数稍微多于CPU核心数
    // 但不超过CPU核心数的2倍，避免过度上下文切换
    let recommended = if cpu_count <= 4 {
        cpu_count
    } else if cpu_count <= 8 {
        cpu_count + 2
    } else {
        (cpu_count as f32 * 1.5) as usize
    };
    
    // 确保至少有1个线程，最多不超过32个线程
    recommended.max(1).min(32)
}

/// 打印CPU信息和线程推荐
fn print_cpu_info() {
    let cpu_count = num_cpus::get();
    let recommended = get_recommended_threads();
    
    println!("CPU信息:");
    println!("  检测到 {} 个CPU核心", cpu_count);
    println!("  推荐线程数: {} 个", recommended);
    println!("  (针对图像处理任务优化)");
    println!();
}

/// 错误类型统计
#[derive(Debug)]
struct ErrorStats {
    invalid_png: AtomicUsize,
    truncated_file: AtomicUsize,
    permission_error: AtomicUsize,
    conversion_error: AtomicUsize,
    other_error: AtomicUsize,
}

impl ErrorStats {
    fn new() -> Self {
        Self {
            invalid_png: AtomicUsize::new(0),
            truncated_file: AtomicUsize::new(0),
            permission_error: AtomicUsize::new(0),
            conversion_error: AtomicUsize::new(0),
            other_error: AtomicUsize::new(0),
        }
    }

    fn record_error(&self, error: &str) {
        if error.contains("Invalid PNG signature") || error.contains("不是有效的PNG格式") {
            self.invalid_png.fetch_add(1, Ordering::Relaxed);
        } else if error.contains("unexpected end of file") || error.contains("截断") {
            self.truncated_file.fetch_add(1, Ordering::Relaxed);
        } else if error.contains("permission") || error.contains("access") || error.contains("权限") {
            self.permission_error.fetch_add(1, Ordering::Relaxed);
        } else if error.contains("conversion") || error.contains("转换失败") {
            self.conversion_error.fetch_add(1, Ordering::Relaxed);
        } else {
            self.other_error.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn print_summary(&self) {
        let invalid = self.invalid_png.load(Ordering::Relaxed);
        let truncated = self.truncated_file.load(Ordering::Relaxed);
        let permission = self.permission_error.load(Ordering::Relaxed);
        let conversion = self.conversion_error.load(Ordering::Relaxed);
        let other = self.other_error.load(Ordering::Relaxed);
        
        if invalid > 0 || truncated > 0 || permission > 0 || conversion > 0 || other > 0 {
            println!("\n错误详情:");
            if invalid > 0 {
                println!("  无效的PNG文件: {} 个", invalid);
            }
            if truncated > 0 {
                println!("  文件被截断/损坏: {} 个", truncated);
            }
            if permission > 0 {
                println!("  权限错误: {} 个", permission);
            }
            if conversion > 0 {
                println!("  转换失败: {} 个", conversion);
            }
            if other > 0 {
                println!("  其他错误: {} 个", other);
            }
        }
    }
}

/// 验证文件是否为有效的PNG文件
fn is_valid_png(file_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let mut file = fs::File::open(file_path)?;
    let mut header = [0u8; 8];
    
    // 读取PNG文件头
    match file.read_exact(&mut header) {
        Ok(_) => {
            // PNG文件的魔数: 89 50 4E 47 0D 0A 1A 0A
            let png_signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            Ok(header == png_signature)
        }
        Err(_) => Ok(false), // 文件太小，不是有效的PNG
    }
}

/// 转换质量设置
#[derive(Debug, Clone, Copy)]
enum CompressionLevel {
    None,
    Fast,
    Balanced,
    Max,
}

/// 将单个 PNG 转换为 TIF
fn convert_png_to_tif(
    input_path: &Path, 
    output_path: &Path,
    _compression: CompressionLevel,
    preserve_transparency: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // 首先验证文件是否为有效的PNG
    if !is_valid_png(input_path)? {
        return Err(format!("文件不是有效的PNG格式: {:?}", input_path).into());
    }

    // 读取PNG图片
    let img = image::open(input_path)?;
    
    // 创建输出目录（如果不存在）
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // 根据是否保留透明度来处理图片并直接保存
    if preserve_transparency && img.color().has_alpha() {
        // 保留透明度，转换为RGBA并保存
        let rgba_img = img.to_rgba8();
        let output_file = fs::File::create(output_path)?;
        let encoder = image::codecs::tiff::TiffEncoder::new(output_file);
        
        encoder.write_image(
            rgba_img.as_raw(),
            rgba_img.width(),
            rgba_img.height(),
            ColorType::Rgba8,
        )?;
    } else {
        // 转换为RGB，透明区域填充白色
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        let mut rgb_img = ImageBuffer::new(width, height);
        
        for (x, y, pixel) in rgba.enumerate_pixels() {
            let Rgba([r, g, b, a]) = *pixel;
            
            if a < 255 {
                // 如果像素有透明度，则根据透明度混合白色背景
                let alpha = a as f32 / 255.0;
                let new_r = (r as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                let new_g = (g as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                let new_b = (b as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                rgb_img.put_pixel(x, y, image::Rgb([new_r, new_g, new_b]));
            } else {
                // 完全不透明的像素直接复制
                rgb_img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        
        let output_file = fs::File::create(output_path)?;
        let encoder = image::codecs::tiff::TiffEncoder::new(output_file);
        
        encoder.write_image(
            rgb_img.as_raw(),
            rgb_img.width(),
            rgb_img.height(),
            ColorType::Rgb8,
        )?;
    }

    Ok(())
}

/// 带重试机制的文件处理
fn convert_png_to_tif_with_retry(
    input_path: &Path, 
    output_path: &Path,
    compression: CompressionLevel,
    preserve_transparency: bool,
    max_retries: usize
) -> Result<(), Box<dyn std::error::Error>> {
    let mut last_error = None;
    
    for attempt in 0..=max_retries {
        match convert_png_to_tif(input_path, output_path, compression, preserve_transparency) {
            Ok(_) => return Ok(()),
            Err(e) => {
                last_error = Some(e);
                
                // 如果是文件访问相关的错误，稍等一下再重试
                if attempt < max_retries {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }
    
    Err(last_error.unwrap())
}

/// 获取所有 PNG 文件路径
fn get_png_files(input_dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

/// 批量处理PNG转TIF
fn batch_process(
    input_dir: &Path,
    output_dir: &Path,
    num_threads: Option<usize>,
    compression: CompressionLevel,
    preserve_transparency: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    // 设置线程池
    let threads = num_threads.unwrap_or_else(get_recommended_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    // 获取所有PNG文件
    println!("正在扫描PNG文件...");
    let png_files = get_png_files(input_dir);
    
    if png_files.is_empty() {
        println!("在输入目录中未找到PNG文件");
        return Ok(());
    }

    println!("找到 {} 个PNG文件", png_files.len());
    println!("使用 {} 个线程进行转换", threads);
    println!("压缩级别: {:?}", compression);
    println!("保留透明度: {}", if preserve_transparency { "是" } else { "否" });
    println!();

    // 创建输出目录
    fs::create_dir_all(output_dir)?;

    // 创建进度条
    let pb = ProgressBar::new(png_files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    // 统计信息
    let processed_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));
    let error_stats = Arc::new(ErrorStats::new());

    // 并行处理所有文件
    png_files.par_iter().for_each(|input_path| {
        // 计算相相对路径
        let relative_path = input_path.strip_prefix(input_dir).unwrap();
        let mut output_path = output_dir.join(relative_path);
        output_path.set_extension("tif");

        match convert_png_to_tif_with_retry(
            input_path, 
            &output_path, 
            compression, 
            preserve_transparency, 
            3
        ) {
            Ok(_) => {
                processed_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                error_count.fetch_add(1, Ordering::Relaxed);
                error_stats.record_error(&e.to_string());
                eprintln!("转换失败 {:?}: {}", input_path, e);
            }
        }
        
        pb.inc(1);
    });

    pb.finish_with_message("转换完成");

    // 打印统计信息
    let processed = processed_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);
    let elapsed = start_time.elapsed();

    println!("\n=== 转换统计 ===");
    println!("总文件数: {}", png_files.len());
    println!("成功转换: {}", processed);
    println!("失败数量: {}", errors);
    println!("总耗时: {:.2} 秒", elapsed.as_secs_f64());
    
    if processed > 0 {
        println!("平均速度: {:.2} 文件/秒", processed as f64 / elapsed.as_secs_f64());
    }

    error_stats.print_summary();

    Ok(())
}

/// 解析命令行参数
fn parse_args() -> ArgMatches {
    Command::new("PNG to TIF 转换器")
        .version("1.0")
        .author("PNG Toolkit")
        .about("高性能PNG转TIF格式转换工具")
        .arg(Arg::new("input")
            .value_name("INPUT_DIR")
            .help("输入目录路径")
            .required_unless_present("info")
            .index(1))
        .arg(Arg::new("output")
            .value_name("OUTPUT_DIR")
            .help("输出目录路径")
            .required_unless_present("info")
            .index(2))
        .arg(Arg::new("threads")
            .short('t')
            .long("threads")
            .value_name("NUMBER|auto")
            .help("并行线程数，可以是数字或 'auto'（默认根据CPU核心数自动选择）"))
        .arg(Arg::new("compression")
            .short('c')
            .long("compression")
            .value_name("LEVEL")
            .help("压缩级别: none, fast, balanced, max")
            .default_value("balanced"))
        .arg(Arg::new("preserve-alpha")
            .long("preserve-alpha")
            .help("保留透明度通道（生成RGBA TIF）")
            .action(clap::ArgAction::SetTrue))
        .arg(Arg::new("info")
            .long("info")
            .help("显示CPU信息")
            .action(clap::ArgAction::SetTrue))
        .get_matches()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = parse_args();

    // 如果需要显示CPU信息
    if matches.get_flag("info") {
        print_cpu_info();
        if !matches.contains_id("input") || !matches.contains_id("output") {
            return Ok(());
        }
    }

    let input_dir = Path::new(matches.get_one::<String>("input").unwrap());
    let output_dir = Path::new(matches.get_one::<String>("output").unwrap());
    
    // 解析线程数参数
    let threads = if let Some(threads_str) = matches.get_one::<String>("threads") {
        if threads_str == "auto" {
            None // 使用自动选择
        } else {
            match threads_str.parse::<usize>() {
                Ok(n) => Some(n),
                Err(_) => {
                    eprintln!("无效的线程数: '{}'，使用自动选择", threads_str);
                    None
                }
            }
        }
    } else {
        None // 用户未指定，使用自动选择
    };
    
    let preserve_transparency = matches.get_flag("preserve-alpha");

    // 解析压缩级别
    let compression = match matches.get_one::<String>("compression").unwrap().as_str() {
        "none" => CompressionLevel::None,
        "fast" => CompressionLevel::Fast,
        "balanced" => CompressionLevel::Balanced,
        "max" => CompressionLevel::Max,
        _ => {
            eprintln!("无效的压缩级别，使用默认值 'balanced'");
            CompressionLevel::Balanced
        }
    };

    // 验证输入目录
    if !input_dir.exists() {
        return Err(format!("输入目录不存在: {:?}", input_dir).into());
    }

    if !input_dir.is_dir() {
        return Err(format!("输入路径不是目录: {:?}", input_dir).into());
    }

    println!("PNG 转 TIF 转换器启动");
    println!("输入目录: {:?}", input_dir);
    println!("输出目录: {:?}", output_dir);
    
    if let Some(t) = threads {
        println!("使用线程数: {}", t);
    } else {
        println!("使用推荐线程数: {}", get_recommended_threads());
    }
    
    println!();

    // 开始批量处理
    batch_process(input_dir, output_dir, threads, compression, preserve_transparency)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_png_to_tif_conversion() {
        let temp_dir = tempdir().unwrap();
        let input_path = temp_dir.path().join("test.png");
        let output_path = temp_dir.path().join("test.tif");

        // 创建一个简单的测试PNG图片
        let img = image::ImageBuffer::from_fn(100, 100, |x, y| {
            if (x + y) % 2 == 0 {
                image::Rgba([255u8, 0, 0, 255]) // 红色
            } else {
                image::Rgba([0u8, 255, 0, 128]) // 半透明绿色
            }
        });
        
        img.save(&input_path).unwrap();

        // 测试转换
        let result = convert_png_to_tif(
            &input_path, 
            &output_path, 
            CompressionLevel::Balanced,
            false
        );

        assert!(result.is_ok());
        assert!(output_path.exists());

        // 验证输出文件是否有效
        let converted_img = image::open(&output_path).unwrap();
        assert_eq!(converted_img.width(), 100);
        assert_eq!(converted_img.height(), 100);
    }

    #[test]
    fn test_compression_levels() {
        let temp_dir = tempdir().unwrap();
        let input_path = temp_dir.path().join("test.png");

        // 创建测试图片
        let img = image::ImageBuffer::from_fn(50, 50, |x, y| {
            image::Rgba([(x % 256) as u8, (y % 256) as u8, 128u8, 255u8])
        });
        img.save(&input_path).unwrap();

        // 测试不同压缩级别
        let compression_levels = [
            CompressionLevel::None,
            CompressionLevel::Fast, 
            CompressionLevel::Balanced,
            CompressionLevel::Max,
        ];

        for (i, compression) in compression_levels.iter().enumerate() {
            let output_path = temp_dir.path().join(format!("test_{}.tif", i));
            let result = convert_png_to_tif(&input_path, &output_path, *compression, false);
            assert!(result.is_ok());
            assert!(output_path.exists());
        }
    }
} 