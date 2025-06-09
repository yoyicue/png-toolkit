use image::{ImageBuffer, Rgba};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::Read;
use wgpu::util::DeviceExt;

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
    other_error: AtomicUsize,
}

impl ErrorStats {
    fn new() -> Self {
        Self {
            invalid_png: AtomicUsize::new(0),
            truncated_file: AtomicUsize::new(0),
            permission_error: AtomicUsize::new(0),
            other_error: AtomicUsize::new(0),
        }
    }

    fn record_error(&self, error: &str) {
        if error.contains("Invalid PNG signature") {
            self.invalid_png.fetch_add(1, Ordering::Relaxed);
        } else if error.contains("unexpected end of file") {
            self.truncated_file.fetch_add(1, Ordering::Relaxed);
        } else if error.contains("permission") || error.contains("access") {
            self.permission_error.fetch_add(1, Ordering::Relaxed);
        } else {
            self.other_error.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn print_summary(&self) {
        let invalid = self.invalid_png.load(Ordering::Relaxed);
        let truncated = self.truncated_file.load(Ordering::Relaxed);
        let permission = self.permission_error.load(Ordering::Relaxed);
        let other = self.other_error.load(Ordering::Relaxed);
        
        if invalid > 0 || truncated > 0 || permission > 0 || other > 0 {
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

/// 带重试机制的文件处理
fn convert_transparent_to_white_with_retry(
    input_path: &Path, 
    output_path: &Path,
    max_retries: usize
) -> Result<(), Box<dyn std::error::Error>> {
    // 首先验证文件是否为有效的PNG
    if !is_valid_png(input_path)? {
        return Err(format!("文件不是有效的PNG格式: {:?}", input_path).into());
    }

    let mut last_error = None;
    
    for attempt in 0..=max_retries {
        match convert_transparent_to_white(input_path, output_path) {
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

/// 处理模式枚举
#[derive(Debug, Clone, Copy)]
enum ProcessingMode {
    Cpu,
    Gpu,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("用法: {} <输入目录> <输出目录> [线程数/模式]", args[0]);
        eprintln!("\n示例:");
        eprintln!("  {} ./input ./output          # CPU模式，自动检测推荐线程数", args[0]);
        eprintln!("  {} ./input ./output 16       # CPU模式，手动指定16个线程", args[0]);
        eprintln!("  {} ./input ./output auto     # CPU模式，显式使用自动检测", args[0]);
        eprintln!("  {} ./input ./output gpu      # GPU模式（推荐）", args[0]);
        std::process::exit(1);
    }
    
    let input_dir = Path::new(&args[1]);
    let output_dir = Path::new(&args[2]);
    
    if !input_dir.exists() {
        eprintln!("错误: 输入目录不存在: {:?}", input_dir);
        std::process::exit(1);
    }
    
    // 解析处理模式和线程数参数
    let (processing_mode, num_threads) = if args.len() > 3 {
        match args[3].as_str() {
            "gpu" => (ProcessingMode::Gpu, None),
            "auto" => (ProcessingMode::Cpu, None), // 显式自动检测
            _ => {
                match args[3].parse::<usize>() {
                    Ok(n) if n > 0 => (ProcessingMode::Cpu, Some(n)),
                    Ok(_) => {
                        eprintln!("错误: 线程数必须大于0");
                        std::process::exit(1);
                    }
                    Err(_) => {
                        eprintln!("错误: 无效的参数: {}", args[3]);
                        eprintln!("请输入一个正整数、'auto' 或 'gpu'");
                        std::process::exit(1);
                    }
                }
            }
        }
    } else {
        (ProcessingMode::Cpu, None) // 默认CPU模式，自动检测线程数
    };
    
    // 显示CPU信息
    print_cpu_info();
    
    // 根据处理模式执行
    match processing_mode {
        ProcessingMode::Cpu => {
            batch_process_cpu(input_dir, output_dir, num_threads)?;
        }
        ProcessingMode::Gpu => {
            println!("正在初始化GPU...");
            match pollster::block_on(batch_process_gpu(input_dir, output_dir)) {
                Ok(_) => {},
                Err(e) => {
                    eprintln!("GPU处理失败: {}", e);
                    eprintln!("回退到CPU模式...");
                    batch_process_cpu(input_dir, output_dir, None)?;
                }
            }
        }
    }
    
    Ok(())
}

/// GPU模式批量处理
async fn batch_process_gpu(
    input_dir: &Path,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // 创建输出目录
    std::fs::create_dir_all(output_dir)?;
    
    // 初始化GPU处理器
    let gpu_processor = GpuProcessor::new().await?;
    
    // 获取所有 PNG 文件
    println!("正在扫描 PNG 文件...");
    let png_files = get_png_files(input_dir);
    let total_files = png_files.len();
    
    if total_files == 0 {
        println!("未找到任何 PNG 文件！");
        return Ok(());
    }
    
    println!("找到 {} 个 PNG 文件", total_files);
    println!("使用GPU加速处理");
    
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
    let error_stats = Arc::new(ErrorStats::new());
    
    let start_time = Instant::now();
    
    // GPU模式下使用较少的并发数，因为GPU处理本身已经高度并行
    rayon::ThreadPoolBuilder::new()
        .num_threads(4) // 限制并发数，避免GPU资源竞争
        .build_global()
        .unwrap();
    
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
        match pollster::block_on(convert_transparent_to_white_gpu(input_path, &output_path, &gpu_processor)) {
            Ok(_) => {
                success_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                let error_msg = e.to_string();
                error_stats.record_error(&error_msg);
                eprintln!("处理 {:?} 时出错: {}", input_path, error_msg);
            }
        }
        
        progress.inc(1);
    });
    
    progress.finish();
    
    let elapsed = start_time.elapsed();
    let success = success_count.load(Ordering::Relaxed);
    let total_errors = total_files - success;
    
    println!("\nGPU处理完成！");
    println!("总耗时: {:.2}s", elapsed.as_secs_f64());
    println!("成功: {} 个文件", success);
    println!("失败: {} 个文件", total_errors);
    println!("平均速度: {:.2} 个文件/秒", total_files as f64 / elapsed.as_secs_f64());
    
    // 打印错误统计
    error_stats.print_summary();
    
    Ok(())
}

/// CPU模式批量处理（重命名原来的batch_process函数）
fn batch_process_cpu(
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
    let actual_threads = match num_threads {
        Some(threads) => {
            println!("使用手动指定的线程数: {}", threads);
            threads
        }
        None => {
            let recommended = get_recommended_threads();
            println!("使用推荐的线程数: {}", recommended);
            recommended
        }
    };
    
    rayon::ThreadPoolBuilder::new()
        .num_threads(actual_threads)
        .build_global()
        .unwrap();
    
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
    let error_stats = Arc::new(ErrorStats::new());
    
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
        
        // 处理文件（带重试机制）
        match convert_transparent_to_white_with_retry(input_path, &output_path, 2) {
            Ok(_) => {
                success_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                let error_msg = e.to_string();
                error_stats.record_error(&error_msg);
                eprintln!("处理 {:?} 时出错: {}", input_path, error_msg);
            }
        }
        
        progress.inc(1);
    });
    
    progress.finish();
    
    let elapsed = start_time.elapsed();
    let success = success_count.load(Ordering::Relaxed);
    let total_errors = total_files - success;
    
    println!("\nCPU处理完成！");
    println!("总耗时: {:.2}s", elapsed.as_secs_f64());
    println!("成功: {} 个文件", success);
    println!("失败: {} 个文件", total_errors);
    println!("平均速度: {:.2} 个文件/秒", total_files as f64 / elapsed.as_secs_f64());
    
    // 打印错误统计
    error_stats.print_summary();
    
    if total_errors > 0 {
        println!("\n建议:");
        println!("- 对于损坏的文件，请检查源文件是否完整");
        println!("- 对于无效的PNG文件，请确认文件格式是否正确");
        println!("- 如果错误持续出现，可以尝试减少并发线程数");
        println!("- 可以尝试使用GPU模式: {} <输入目录> <输出目录> gpu", std::env::args().nth(0).unwrap_or_default());
    }
    
    Ok(())
}

/// GPU处理器结构
struct GpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuProcessor {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // 初始化wgpu实例
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL, // 在macOS上使用Metal
            ..Default::default()
        });

        // 获取适配器（优先选择高性能GPU）
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("未找到适合的GPU适配器")?;

        println!("GPU信息: {}", adapter.get_info().name);
        
        // 获取设备和队列
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await?;

        // 加载shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("透明度处理 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("transparency_shader.wgsl").into()),
        });

        // 创建绑定组布局
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("透明度处理绑定组布局"),
        });

        // 创建计算管线
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("透明度处理管线布局"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("透明度处理计算管线"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            compute_pipeline,
            bind_group_layout,
        })
    }

    fn process_image(&self, rgba_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let pixel_count = (width * height) as usize;
        
        // 将RGBA字节转换为u32数组（每个像素一个u32）
        let mut input_pixels: Vec<u32> = Vec::with_capacity(pixel_count);
        for chunk in rgba_data.chunks_exact(4) {
            let pixel = ((chunk[0] as u32) << 24) | 
                       ((chunk[1] as u32) << 16) | 
                       ((chunk[2] as u32) << 8) | 
                       (chunk[3] as u32);
            input_pixels.push(pixel);
        }

        // 创建GPU缓冲区
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("输入缓冲区"),
            contents: bytemuck::cast_slice(&input_pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("输出缓冲区"),
            size: (pixel_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 创建用于读取结果的缓冲区
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("暂存缓冲区"),
            size: (pixel_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 创建uniform缓冲区传递图像尺寸
        let dimensions_data = [width, height];
        let dimensions_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("尺寸缓冲区"),
            contents: bytemuck::cast_slice(&dimensions_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 创建绑定组
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dimensions_buffer.as_entire_binding(),
                },
            ],
            label: Some("透明度处理绑定组"),
        });

        // 创建命令编码器
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("透明度处理命令编码器"),
        });

        // 计算工作组数量
        let workgroup_size = 16;
        let workgroups_x = (width + workgroup_size - 1) / workgroup_size;
        let workgroups_y = (height + workgroup_size - 1) / workgroup_size;

        // 分发计算着色器
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("透明度处理计算通道"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // 复制结果到暂存缓冲区
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (pixel_count * 4) as u64);

        // 提交命令
        self.queue.submit(std::iter::once(encoder.finish()));

        // 等待GPU完成并读取结果
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        pollster::block_on(receiver.receive()).unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result_pixels: &[u32] = bytemuck::cast_slice(&data);
        
        // 将u32像素转换回RGBA字节
        let mut result_bytes = Vec::with_capacity(pixel_count * 4);
        for &pixel in result_pixels {
            result_bytes.push((pixel >> 24) as u8); // R
            result_bytes.push((pixel >> 16) as u8); // G
            result_bytes.push((pixel >> 8) as u8);  // B
            result_bytes.push(pixel as u8);         // A
        }

        drop(data);
        staging_buffer.unmap();

        Ok(result_bytes)
    }
}

/// 使用GPU处理图像
async fn convert_transparent_to_white_gpu(
    input_path: &Path, 
    output_path: &Path,
    gpu_processor: &GpuProcessor
) -> Result<(), Box<dyn std::error::Error>> {
    // 首先验证文件是否为有效的PNG
    if !is_valid_png(input_path)? {
        return Err(format!("文件不是有效的PNG格式: {:?}", input_path).into());
    }

    // 读取图片
    let img = image::open(input_path)?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    // 使用GPU处理
    let processed_data = gpu_processor.process_image(&rgba, width, height)?;
    
    // 创建新的图片并保存
    let new_img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, processed_data)
        .ok_or("无法创建图像缓冲区")?;
    
    new_img.save(output_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};
    
    #[test]
    fn test_transparent_to_white_conversion() {
        // 创建一个测试图片
        let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(2, 2);
        
        // 设置像素：
        // (0,0): 完全透明的红色
        // (0,1): 半透明的绿色
        // (1,0): 不透明的蓝色
        // (1,1): 完全透明
        img.put_pixel(0, 0, Rgba([255u8, 0u8, 0u8, 0u8]));
        img.put_pixel(0, 1, Rgba([0u8, 255u8, 0u8, 128u8]));
        img.put_pixel(1, 0, Rgba([0u8, 0u8, 255u8, 255u8]));
        img.put_pixel(1, 1, Rgba([0u8, 0u8, 0u8, 0u8]));
        
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
        assert_eq!(*result.get_pixel(0, 1), Rgba([126, 255, 126, 255])); // 混合后的颜色 (128/255 * 0 + 255 * (1 - 128/255))
        assert_eq!(*result.get_pixel(1, 0), Rgba([0, 0, 255, 255]));     // 蓝色不变
        assert_eq!(*result.get_pixel(1, 1), Rgba([255, 255, 255, 255])); // 完全白色
        
        // 清理测试文件
        std::fs::remove_file(test_input).unwrap();
        std::fs::remove_file(test_output).unwrap();
    }
}