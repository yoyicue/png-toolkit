use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;

/// 过滤模式
#[derive(Debug, Clone)]
enum FilterMode {
    /// 白名单模式：只保留包含指定关键词的文件
    Whitelist(Vec<String>),
    /// 黑名单模式：删除包含指定关键词的文件
    Blacklist(Vec<String>),
}

/// 获取推荐的线程数
fn get_recommended_threads() -> usize {
    let cpu_count = num_cpus::get();
    
    // 对于文件删除任务，I/O密集型，建议线程数稍多于CPU核心数
    let recommended = if cpu_count <= 4 {
        cpu_count * 2
    } else if cpu_count <= 8 {
        cpu_count + 4
    } else {
        (cpu_count as f32 * 1.8) as usize
    };
    
    // 确保至少有2个线程，最多不超过64个线程
    recommended.max(2).min(64)
}

/// 打印CPU信息和线程推荐
fn print_cpu_info() {
    let cpu_count = num_cpus::get();
    let recommended = get_recommended_threads();
    
    println!("CPU信息:");
    println!("  检测到 {} 个CPU核心", cpu_count);
    println!("  推荐线程数: {} 个", recommended);
    println!("  (针对文件删除任务优化)");
    println!();
}

/// 错误统计
#[derive(Debug)]
struct ErrorStats {
    permission_error: AtomicUsize,
    file_not_found: AtomicUsize,
    other_error: AtomicUsize,
}

impl ErrorStats {
    fn new() -> Self {
        Self {
            permission_error: AtomicUsize::new(0),
            file_not_found: AtomicUsize::new(0),
            other_error: AtomicUsize::new(0),
        }
    }

    fn record_error(&self, error: &str) {
        if error.contains("permission") || error.contains("access") || error.contains("denied") {
            self.permission_error.fetch_add(1, Ordering::Relaxed);
        } else if error.contains("not found") || error.contains("No such file") {
            self.file_not_found.fetch_add(1, Ordering::Relaxed);
        } else {
            self.other_error.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn print_summary(&self) {
        let permission = self.permission_error.load(Ordering::Relaxed);
        let not_found = self.file_not_found.load(Ordering::Relaxed);
        let other = self.other_error.load(Ordering::Relaxed);
        
        if permission > 0 || not_found > 0 || other > 0 {
            println!("\n错误详情:");
            if permission > 0 {
                println!("  权限错误: {} 个", permission);
            }
            if not_found > 0 {
                println!("  文件未找到: {} 个", not_found);
            }
            if other > 0 {
                println!("  其他错误: {} 个", other);
            }
        }
    }
}

/// 检查文件是否应该被保留
fn should_keep_file(file_path: &Path, filter_mode: &FilterMode) -> bool {
    if let Some(filename) = file_path.file_name().and_then(|n| n.to_str()) {
        match filter_mode {
            FilterMode::Whitelist(keywords) => {
                // 白名单模式：文件名包含任意关键词则保留
                keywords.iter().any(|keyword| filename.contains(keyword))
            }
            FilterMode::Blacklist(keywords) => {
                // 黑名单模式：文件名不包含任何关键词则保留
                !keywords.iter().any(|keyword| filename.contains(keyword))
            }
        }
    } else {
        false // 如果无法获取文件名，默认不保留
    }
}

/// 获取所有需要检查的PNG文件路径
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

/// 带重试机制的文件删除
fn delete_file_with_retry(file_path: &Path, max_retries: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut last_error = None;
    
    for attempt in 0..=max_retries {
        match fs::remove_file(file_path) {
            Ok(_) => return Ok(()),
            Err(e) => {
                last_error = Some(e);
                
                // 如果是文件访问相关的错误，稍等一下再重试
                if attempt < max_retries {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
        }
    }
    
    Err(last_error.unwrap().into())
}

/// 批量删除不符合条件的PNG文件
fn batch_delete_png_files(
    target_dir: &Path,
    filter_mode: FilterMode,
    num_threads: Option<usize>,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !target_dir.exists() {
        return Err(format!("目标目录不存在: {:?}", target_dir).into());
    }
    
    // 获取所有PNG文件
    println!("正在扫描PNG文件...");
    let png_files = get_png_files(target_dir);
    let total_files = png_files.len();
    
    if total_files == 0 {
        println!("未找到任何PNG文件！");
        return Ok(());
    }
    
    println!("找到 {} 个PNG文件", total_files);
    
    // 过滤出需要删除的文件
    let files_to_delete: Vec<PathBuf> = png_files
        .into_iter()
        .filter(|path| !should_keep_file(path, &filter_mode))
        .collect();
    
    let delete_count = files_to_delete.len();
    let keep_count = total_files - delete_count;
    
    println!("\n文件分析结果:");
    println!("  需要保留: {} 个文件", keep_count);
    println!("  需要删除: {} 个文件", delete_count);
    
    if delete_count == 0 {
        println!("没有需要删除的文件！");
        return Ok(());
    }
    
    if dry_run {
        println!("\n=== 试运行模式 - 以下文件将被删除 ===");
        for (i, file) in files_to_delete.iter().enumerate() {
            println!("{}. {:?}", i + 1, file);
            if i >= 20 {
                println!("... 还有 {} 个文件", delete_count - 21);
                break;
            }
        }
        return Ok(());
    }
    
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
    let progress = Arc::new(ProgressBar::new(delete_count as u64));
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.red/red} {pos}/{len} ({percent}%) 删除中...")
            .unwrap()
            .progress_chars("##-"),
    );
    
    // 统计成功和失败的数量
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_stats = Arc::new(ErrorStats::new());
    
    let start_time = Instant::now();
    
    // 并行删除文件
    files_to_delete.par_iter().for_each(|file_path| {
        match delete_file_with_retry(file_path, 2) {
            Ok(_) => {
                success_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                let error_msg = e.to_string();
                error_stats.record_error(&error_msg);
                eprintln!("删除 {:?} 时出错: {}", file_path, error_msg);
            }
        }
        
        progress.inc(1);
    });
    
    progress.finish();
    
    let elapsed = start_time.elapsed();
    let success = success_count.load(Ordering::Relaxed);
    let total_errors = delete_count - success;
    
    println!("\n删除操作完成！");
    println!("总耗时: {:.2}s", elapsed.as_secs_f64());
    println!("成功删除: {} 个文件", success);
    println!("删除失败: {} 个文件", total_errors);
    if delete_count > 0 {
        println!("平均速度: {:.2} 个文件/秒", delete_count as f64 / elapsed.as_secs_f64());
    }
    
    // 打印错误统计
    error_stats.print_summary();
    
    if total_errors > 0 {
        println!("\n建议:");
        println!("- 对于权限错误，请检查是否有足够的文件删除权限");
        println!("- 对于文件未找到错误，可能是文件在处理过程中被其他程序删除");
        println!("- 如果错误持续出现，可以尝试减少并发线程数");
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("用法: {} <目标目录> [过滤选项] [其他选项]", args[0]);
        eprintln!("\n过滤选项（必须指定一个）:");
        eprintln!("  --whitelist <关键词1,关键词2,...>  白名单模式：保留包含指定关键词的文件");
        eprintln!("  --blacklist <关键词1,关键词2,...>  黑名单模式：删除包含指定关键词的文件");
        eprintln!("\n其他选项:");
        eprintln!("  --threads <数量>                   手动指定线程数");
        eprintln!("  --dry-run                          试运行模式，只显示将要删除的文件");
        eprintln!("  --auto                             使用自动检测的推荐线程数（默认）");
        eprintln!("\n示例:");
        eprintln!("  # 白名单模式：保留包含这些关键词的文件");
        eprintln!("  {} ./images --whitelist \"PreAdvanced,Intermediate,Essentials,EasyEssentials\"", args[0]);
        eprintln!("  # 黑名单模式：删除包含这些关键词的文件");
        eprintln!("  {} ./images --blacklist \"temp,cache,backup\"", args[0]);
        eprintln!("  # 先试运行查看效果");
        eprintln!("  {} ./images --whitelist \"Essential\" --dry-run", args[0]);
        std::process::exit(1);
    }
    
    let target_dir = Path::new(&args[1]);
    let mut filter_mode: Option<FilterMode> = None;
    let mut num_threads = None;
    let mut dry_run = false;
    
    // 解析命令行参数
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--whitelist" => {
                if i + 1 < args.len() {
                    let keywords: Vec<String> = args[i + 1]
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    
                    if keywords.is_empty() {
                        eprintln!("错误: 白名单关键词不能为空");
                        std::process::exit(1);
                    }
                    
                    filter_mode = Some(FilterMode::Whitelist(keywords));
                    i += 2;
                } else {
                    eprintln!("错误: --whitelist 选项需要关键词参数");
                    std::process::exit(1);
                }
            }
            "--blacklist" => {
                if i + 1 < args.len() {
                    let keywords: Vec<String> = args[i + 1]
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    
                    if keywords.is_empty() {
                        eprintln!("错误: 黑名单关键词不能为空");
                        std::process::exit(1);
                    }
                    
                    filter_mode = Some(FilterMode::Blacklist(keywords));
                    i += 2;
                } else {
                    eprintln!("错误: --blacklist 选项需要关键词参数");
                    std::process::exit(1);
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<usize>() {
                        Ok(n) if n > 0 => {
                            num_threads = Some(n);
                            i += 2;
                        }
                        Ok(_) => {
                            eprintln!("错误: 线程数必须大于0");
                            std::process::exit(1);
                        }
                        Err(_) => {
                            eprintln!("错误: 无效的线程数: {}", args[i + 1]);
                            std::process::exit(1);
                        }
                    }
                } else {
                    eprintln!("错误: --threads 选项需要一个数值参数");
                    std::process::exit(1);
                }
            }
            "--dry-run" => {
                dry_run = true;
                i += 1;
            }
            "--auto" => {
                num_threads = None;
                i += 1;
            }
            _ => {
                eprintln!("错误: 未知的选项: {}", args[i]);
                std::process::exit(1);
            }
        }
    }
    
    // 检查是否指定了过滤模式
    let filter_mode = match filter_mode {
        Some(mode) => mode,
        None => {
            eprintln!("错误: 必须指定 --whitelist 或 --blacklist 选项");
            std::process::exit(1);
        }
    };
    
    if !target_dir.exists() {
        eprintln!("错误: 目标目录不存在: {:?}", target_dir);
        std::process::exit(1);
    }
    
    // 显示CPU信息
    print_cpu_info();
    
    // 显示过滤规则
    match &filter_mode {
        FilterMode::Whitelist(keywords) => {
            println!("过滤模式: 白名单");
            println!("  文件名包含以下任意关键词的PNG文件将被保留:");
            for keyword in keywords {
                println!("  - {}", keyword);
            }
        }
        FilterMode::Blacklist(keywords) => {
            println!("过滤模式: 黑名单");
            println!("  文件名包含以下任意关键词的PNG文件将被删除:");
            for keyword in keywords {
                println!("  - {}", keyword);
            }
        }
    }
    println!();
    
    if dry_run {
        println!("=== 试运行模式 ===");
        println!("只会显示将要删除的文件，不会实际删除");
        println!();
    } else {
        println!("⚠️  警告: 此操作将永久删除文件！");
        println!("   建议先使用 --dry-run 选项预览要删除的文件");
        
        print!("是否确认继续？[y/N]: ");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        if !input.trim().eq_ignore_ascii_case("y") && !input.trim().eq_ignore_ascii_case("yes") {
            println!("操作已取消");
            return Ok(());
        }
        println!();
    }
    
    // 执行删除操作
    batch_delete_png_files(target_dir, filter_mode, num_threads, dry_run)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;
    
    #[test]
    fn test_should_keep_file() {
        // 测试白名单模式
        let whitelist = FilterMode::Whitelist(vec![
            "PreAdvanced".to_string(),
            "Intermediate".to_string(),
            "Essentials".to_string(),
            "EasyEssentials".to_string(),
        ]);
        
        // 测试应该保留的文件
        assert!(should_keep_file(Path::new("test_PreAdvanced.png"), &whitelist));
        assert!(should_keep_file(Path::new("some_Intermediate_file.png"), &whitelist));
        assert!(should_keep_file(Path::new("Essentials_guide.png"), &whitelist));
        assert!(should_keep_file(Path::new("my_EasyEssentials.png"), &whitelist));
        
        // 测试应该删除的文件
        assert!(!should_keep_file(Path::new("regular_file.png"), &whitelist));
        assert!(!should_keep_file(Path::new("test.png"), &whitelist));
        assert!(!should_keep_file(Path::new("advanced.png"), &whitelist)); // 不是 PreAdvanced
        
        // 测试大小写敏感
        assert!(!should_keep_file(Path::new("preadvanced.png"), &whitelist)); // 小写不匹配
        
        // 测试黑名单模式
        let blacklist = FilterMode::Blacklist(vec![
            "temp".to_string(),
            "cache".to_string(),
            "backup".to_string(),
        ]);
        
        // 测试应该保留的文件（不包含黑名单关键词）
        assert!(should_keep_file(Path::new("normal_file.png"), &blacklist));
        assert!(should_keep_file(Path::new("important_data.png"), &blacklist));
        
        // 测试应该删除的文件（包含黑名单关键词）
        assert!(!should_keep_file(Path::new("temp_file.png"), &blacklist));
        assert!(!should_keep_file(Path::new("cache_data.png"), &blacklist));
        assert!(!should_keep_file(Path::new("backup_copy.png"), &blacklist));
    }
    
    #[test]
    fn test_file_filtering() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        
        // 创建测试文件
        let files_to_create = [
            "keep_PreAdvanced.png",
            "keep_Intermediate.png", 
            "keep_Essentials.png",
            "keep_EasyEssentials.png",
            "delete_me.png",
            "another_delete.png",
        ];
        
        for filename in &files_to_create {
            let file_path = temp_path.join(filename);
            let mut file = File::create(&file_path).unwrap();
            // 写入PNG文件头以确保被识别为PNG
            file.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]).unwrap();
        }
        
        // 获取PNG文件
        let png_files = get_png_files(temp_path);
        assert_eq!(png_files.len(), 6);
        
        // 测试白名单过滤逻辑
        let whitelist = FilterMode::Whitelist(vec![
            "PreAdvanced".to_string(),
            "Intermediate".to_string(),
            "Essentials".to_string(),
            "EasyEssentials".to_string(),
        ]);
        
        let files_to_delete: Vec<_> = png_files
            .clone()
            .into_iter()
            .filter(|path| !should_keep_file(path, &whitelist))
            .collect();
        
        assert_eq!(files_to_delete.len(), 2);
        
        // 验证要删除的文件名
        let delete_names: Vec<_> = files_to_delete
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        
        assert!(delete_names.contains(&"delete_me.png"));
        assert!(delete_names.contains(&"another_delete.png"));
        
        // 测试黑名单过滤逻辑
        let blacklist = FilterMode::Blacklist(vec!["delete".to_string()]);
        
        let files_to_delete_blacklist: Vec<_> = png_files
            .into_iter()
            .filter(|path| !should_keep_file(path, &blacklist))
            .collect();
        
        assert_eq!(files_to_delete_blacklist.len(), 2);
        
        // 验证黑名单模式下要删除的文件名
        let delete_names_blacklist: Vec<_> = files_to_delete_blacklist
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        
        assert!(delete_names_blacklist.contains(&"delete_me.png"));
        assert!(delete_names_blacklist.contains(&"another_delete.png"));
    }
} 