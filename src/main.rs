use image::{ImageBuffer, Rgba};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;
use indicatif::{ProgressBar, ProgressStyle};

/// 将单个 PNG 图片的透明背景转换为白色
fn convert_transparent_to_white(input_path: &Path, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // 读取图片
    let img = image::open(input_path)?;
    
    // 转换为 RGBA8 格式
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    // 创建新的图片缓冲区
    let mut new_img = ImageBuffer::new(width, height);
    
    // 遍历每个像素
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let Rgba([r, g, b, a]) = *pixel;
        
        if a < 255 {
            // 如果像素有透明度，则根据透明度混合白色背景
            let alpha = a as f32 / 255.0;
            let new_r = (r as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            let new_g = (g as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            let new_b = (b as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            new_img.put_pixel(x, y, Rgba([new_r, new_g, new_b, 255]));
        } else {
            // 完全不透明的像素直接复制
            new_img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    
    // 保存为 PNG
    new_img.save(output_path)?;
    Ok(())
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

/// 批量处理 PNG 文件
fn batch_process(
    input_dir: &Path,
    output_dir: &Path,
    num_threads: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    // 创建输出目录
    std::fs::create_dir_all(output_dir)?;
    
    // 获取所有 PNG 文件
    println!("正在扫描 PNG 文件...");
    let png_files = get_png_files(input_dir);
    let total_files = png_files.len();
    
    if total_files == 0 {
        println!("未找到任何 PNG 文件！");
        return Ok(());
    }
    
    println!("找到 {} 个 PNG 文件", total_files);
    
    // 设置线程池大小
    if let Some(threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }
    
    // 创建进度条
    let progress = Arc::new(ProgressBar::new(total_files as u64));
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%) {msg}")
            .unwrap()
            .progress_chars("##-"),
    );
    
    // 统计成功和失败的数量
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));
    
    let start_time = Instant::now();
    
    // 并行处理文件
    png_files.par_iter().for_each(|input_path| {
        // 构建输出路径
        let relative_path = input_path.strip_prefix(input_dir).unwrap();
        let output_path = output_dir.join(relative_path);
        
        // 创建输出文件的父目录
        if let Some(parent) = output_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        
        // 处理文件
        match convert_transparent_to_white(input_path, &output_path) {
            Ok(_) => {
                success_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                error_count.fetch_add(1, Ordering::Relaxed);
                eprintln!("处理 {:?} 时出错: {}", input_path, e);
            }
        }
        
        progress.inc(1);
    });
    
    progress.finish();
    
    let elapsed = start_time.elapsed();
    let success = success_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);
    
    println!("\n处理完成！");
    println!("总耗时: {:.2}s", elapsed.as_secs_f64());
    println!("成功: {} 个文件", success);
    println!("失败: {} 个文件", errors);
    println!("平均速度: {:.2} 个文件/秒", total_files as f64 / elapsed.as_secs_f64());
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("用法: {} <输入目录> <输出目录> [线程数]", args[0]);
        eprintln!("\n示例:");
        eprintln!("  {} ./input ./output", args[0]);
        eprintln!("  {} ./input ./output 16", args[0]);
        std::process::exit(1);
    }
    
    let input_dir = Path::new(&args[1]);
    let output_dir = Path::new(&args[2]);
    
    if !input_dir.exists() {
        eprintln!("错误: 输入目录不存在: {:?}", input_dir);
        std::process::exit(1);
    }
    
    let num_threads = if args.len() > 3 {
        args[3].parse::<usize>().ok()
    } else {
        None
    };
    
    batch_process(input_dir, output_dir, num_threads)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};
    
    #[test]
    fn test_transparent_to_white_conversion() {
        // 创建一个测试图片
        let mut img = ImageBuffer::new(2, 2);
        
        // 设置像素：
        // (0,0): 完全透明的红色
        // (0,1): 半透明的绿色
        // (1,0): 不透明的蓝色
        // (1,1): 完全透明
        img.put_pixel(0, 0, Rgba([255, 0, 0, 0]));
        img.put_pixel(0, 1, Rgba([0, 255, 0, 128]));
        img.put_pixel(1, 0, Rgba([0, 0, 255, 255]));
        img.put_pixel(1, 1, Rgba([0, 0, 0, 0]));
        
        // 保存测试输入
        let test_input = "test_input.png";
        let test_output = "test_output.png";
        img.save(test_input).unwrap();
        
        // 执行转换
        convert_transparent_to_white(Path::new(test_input), Path::new(test_output)).unwrap();
        
        // 读取结果
        let result = image::open(test_output).unwrap().to_rgba8();
        
        // 验证结果
        assert_eq!(*result.get_pixel(0, 0), Rgba([255, 255, 255, 255])); // 完全白色
        assert_eq!(*result.get_pixel(0, 1), Rgba([127, 255, 127, 255])); // 混合后的颜色
        assert_eq!(*result.get_pixel(1, 0), Rgba([0, 0, 255, 255]));     // 蓝色不变
        assert_eq!(*result.get_pixel(1, 1), Rgba([255, 255, 255, 255])); // 完全白色
        
        // 清理测试文件
        std::fs::remove_file(test_input).unwrap();
        std::fs::remove_file(test_output).unwrap();
    }
}