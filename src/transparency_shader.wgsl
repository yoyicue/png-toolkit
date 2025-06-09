// 透明背景转换 Compute Shader
// 将透明像素与白色背景混合

@group(0) @binding(0)
var<storage, read> input_data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<u32>;

@group(0) @binding(2)
var<uniform> dimensions: vec2<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // 检查是否在图像范围内
    if x >= dimensions.x || y >= dimensions.y {
        return;
    }
    
    let index = y * dimensions.x + x;
    
    if index >= arrayLength(&input_data) {
        return;
    }
    
    let pixel = input_data[index];
    
    // 解析RGBA值 (假设是RGBA8格式)
    let r = (pixel >> 24u) & 0xFFu;
    let g = (pixel >> 16u) & 0xFFu;
    let b = (pixel >> 8u) & 0xFFu;
    let a = pixel & 0xFFu;
    
    // 如果完全不透明，直接复制
    if a == 255u {
        output_data[index] = (r << 24u) | (g << 16u) | (b << 8u) | 255u;
        return;
    }
    
    // Alpha混合计算：color = src_color * alpha + bg_color * (1 - alpha)
    // 背景色为白色 (255, 255, 255)
    let alpha_f = f32(a) / 255.0;
    let inv_alpha = 1.0 - alpha_f;
    
    let new_r = u32(f32(r) * alpha_f + 255.0 * inv_alpha);
    let new_g = u32(f32(g) * alpha_f + 255.0 * inv_alpha);
    let new_b = u32(f32(b) * alpha_f + 255.0 * inv_alpha);
    
    // 输出完全不透明的像素
    output_data[index] = (new_r << 24u) | (new_g << 16u) | (new_b << 8u) | 255u;
} 