import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import argparse

def setup_chinese_font():
    """
    设置中文字体支持
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"字体设置警告: {e}")

def preprocess_image(image_path):
    """
    图像预处理函数
    """
    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为RGB格式用于显示
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 1. 高斯滤波去噪
    denoised_image = cv2.GaussianBlur(original_image, (5, 5), 0)
    denoised_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    
    # 2. 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2HSV)
    
    # 3. 转换为Lab颜色空间
    lab_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2LAB)
    
    # 4. 对比度增强 - 使用CLAHE
    lab_planes = list(cv2.split(lab_image))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    enhanced_lab = cv2.merge(lab_planes)
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    
    return {
        'original': image_rgb,
        'denoised': denoised_rgb,
        'hsv': hsv_image,
        'lab': lab_image,
        'enhanced': enhanced_rgb,
        'enhanced_bgr': enhanced_bgr
    }

def filter_paper_disks(white_mask, min_area=300, max_area=5000, circularity_threshold=0.7, min_diameter=56):
    """
    筛选真正的滤纸片，排除培养皿边缘等干扰
    """
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    paper_disks = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 面积筛选
        if area < min_area or area > max_area:
            continue
            
        # 圆形度筛选
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < circularity_threshold:
            continue
        
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        diameter = 2 * radius
        
        # 直径筛选：滤纸片直径大于30像素
        if diameter < min_diameter:
            continue
            
        paper_disks.append({
            'contour': contour,
            'center': center,
            'radius': radius,
            'diameter': diameter,
            'area': area,
            'circularity': circularity
        })
    
    # 按面积排序
    paper_disks.sort(key=lambda x: x['area'], reverse=True)
    
    print(f"找到 {len(paper_disks)} 个可能的滤纸片 (直径 > {min_diameter}像素)")
    for i, disk in enumerate(paper_disks):
        print(f"滤纸片 {i+1}: 中心{disk['center']}, 半径{disk['radius']}px, 直径{disk['diameter']}px, 面积{disk['area']:.1f}, 圆形度{disk['circularity']:.3f}")
    
    return paper_disks

def region_segmentation(processed_data):
    """
    区域分割函数 - 新策略：宽范围捕捉 + 透明度过滤
    """
    enhanced_bgr = processed_data['enhanced_bgr']
    enhanced_rgb = processed_data['enhanced']
    hsv_image = processed_data['hsv']
    lab_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
    
    print("新策略：宽范围捕捉所有黄色区域，再过滤透明区域...")
    
    # 第一阶段：宽范围捕捉所有黄色区域（基于你反馈的range345）
    print("第一阶段：宽范围黄色区域捕捉")
    
    # Range 3：宽范围黄色
    lower_yellow_wide = np.array([10, 20, 100])
    upper_yellow_wide = np.array([50, 100, 220])
    mask_yellow_wide = cv2.inRange(hsv_image, lower_yellow_wide, upper_yellow_wide)
    
    # Range 4：中等范围黄色
    lower_yellow_medium = np.array([18, 40, 150])
    upper_yellow_medium = np.array([33, 70, 230])
    mask_yellow_medium = cv2.inRange(hsv_image, lower_yellow_medium, upper_yellow_medium)
    
    # Range 5：较严格黄色
    lower_yellow_strict = np.array([20, 45, 160])
    upper_yellow_strict = np.array([32, 65, 210])
    mask_yellow_strict = cv2.inRange(hsv_image, lower_yellow_strict, upper_yellow_strict)
    
    # 合并所有黄色区域
    all_yellow_mask = cv2.bitwise_or(mask_yellow_wide, mask_yellow_medium)
    all_yellow_mask = cv2.bitwise_or(all_yellow_mask, mask_yellow_strict)
    
    # 形态学操作填充小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    all_yellow_mask = cv2.morphologyEx(all_yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    print(f"宽范围黄色区域统计:")
    print(f"  Range3: {np.count_nonzero(mask_yellow_wide)} 像素")
    print(f"  Range4: {np.count_nonzero(mask_yellow_medium)} 像素")
    print(f"  Range5: {np.count_nonzero(mask_yellow_strict)} 像素")
    print(f"  合并后: {np.count_nonzero(all_yellow_mask)} 像素")
    
    # 第二阶段：过滤掉透明区域（抑菌圈）
    print("第二阶段：过滤透明区域")
    
    # 方法1：基于亮度阈值 - 抑菌圈通常较暗
    gray_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    
    # 在黄色区域内计算亮度特征
    yellow_region_brightness = cv2.bitwise_and(gray_image, gray_image, mask=all_yellow_mask)
    
    # 计算黄色区域的亮度统计
    yellow_pixels = yellow_region_brightness[all_yellow_mask > 0]
    if len(yellow_pixels) > 0:
        brightness_median = np.median(yellow_pixels)
        brightness_std = np.std(yellow_pixels)
        brightness_threshold = brightness_median - 0.5 * brightness_std
        
        print(f"黄色区域亮度统计: 中位数={brightness_median:.1f}, 标准差={brightness_std:.1f}")
        print(f"亮度阈值: {brightness_threshold:.1f}")
        
        # 创建亮度掩码 - 保留较亮的区域（大肠杆菌），过滤较暗的区域（抑菌圈）
        brightness_mask = np.zeros_like(gray_image)
        brightness_mask[gray_image > brightness_threshold] = 255
        
        # 结合黄色区域和亮度掩码
        e_coli_mask = cv2.bitwise_and(all_yellow_mask, brightness_mask)
    else:
        e_coli_mask = all_yellow_mask
        print("警告：未找到黄色区域，使用原始掩码")
    
    # 方法2：基于Lab颜色空间的亮度通道
    lab_L = lab_image[:,:,0]  # 亮度通道
    lab_L_yellow = cv2.bitwise_and(lab_L, lab_L, mask=all_yellow_mask)
    lab_pixels = lab_L_yellow[all_yellow_mask > 0]
    
    if len(lab_pixels) > 0:
        lab_median = np.median(lab_pixels)
        lab_std = np.std(lab_pixels)
        lab_threshold = lab_median - 0.3 * lab_std
        
        lab_brightness_mask = np.zeros_like(lab_L)
        lab_brightness_mask[lab_L > lab_threshold] = 255
        
        # 与HSV方法的结果合并
        e_coli_mask_lab = cv2.bitwise_and(all_yellow_mask, lab_brightness_mask)
        e_coli_mask = cv2.bitwise_or(e_coli_mask, e_coli_mask_lab)
    
    # 最终形态学优化
    e_coli_mask = cv2.morphologyEx(e_coli_mask, cv2.MORPH_CLOSE, kernel)
    e_coli_mask = cv2.morphologyEx(e_coli_mask, cv2.MORPH_OPEN, kernel)
    
    # 显示中间结果
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 6, 1)
    plt.imshow(mask_yellow_wide, cmap='gray')
    plt.title('Range3: Wide Yellow')
    plt.axis('off')
    
    plt.subplot(1, 6, 2)
    plt.imshow(mask_yellow_medium, cmap='gray')
    plt.title('Range4: Medium Yellow')
    plt.axis('off')
    
    plt.subplot(1, 6, 3)
    plt.imshow(mask_yellow_strict, cmap='gray')
    plt.title('Range5: Strict Yellow')
    plt.axis('off')
    
    plt.subplot(1, 6, 4)
    plt.imshow(all_yellow_mask, cmap='gray')
    plt.title('All Yellow Regions')
    plt.axis('off')
    
    plt.subplot(1, 6, 5)
    plt.imshow(e_coli_mask, cmap='gray')
    plt.title('After Transparency Filter')
    plt.axis('off')
    
    # 显示被过滤掉的区域（抑菌圈）
    filtered_out_mask = cv2.bitwise_xor(all_yellow_mask, e_coli_mask)
    plt.subplot(1, 6, 6)
    plt.imshow(filtered_out_mask, cmap='gray')
    plt.title('Filtered Out (Inhibition)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"过滤后统计:")
    print(f"  过滤前: {np.count_nonzero(all_yellow_mask)} 像素")
    print(f"  过滤后: {np.count_nonzero(e_coli_mask)} 像素") 
    print(f"  过滤掉: {np.count_nonzero(filtered_out_mask)} 像素 (抑菌圈区域)")
    
    # 提取最终的大肠杆菌区域
    e_coli_region = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=e_coli_mask)
    e_coli_region_rgb = cv2.cvtColor(e_coli_region, cv2.COLOR_BGR2RGB)
    
    # 白色区域分割（滤纸片）- 保持不变
    print("进行白色区域分割...")
    lower_white_hsv = np.array([0, 0, 200])
    upper_white_hsv = np.array([180, 30, 255])
    white_mask_hsv = cv2.inRange(hsv_image, lower_white_hsv, upper_white_hsv)
    
    lower_white_rgb = np.array([200, 200, 200])
    upper_white_rgb = np.array([255, 255, 255])
    white_mask_rgb = cv2.inRange(enhanced_rgb, lower_white_rgb, upper_white_rgb)
    
    white_mask = cv2.bitwise_or(white_mask_hsv, white_mask_rgb)
    
    kernel_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_white)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_white)
    
    # 筛选滤纸片
    paper_disks = filter_paper_disks(white_mask, min_area=50, max_area=10000, 
                                   circularity_threshold=0.4, min_diameter=15)
    
    paper_mask = np.zeros_like(white_mask)
    for disk in paper_disks:
        cv2.drawContours(paper_mask, [disk['contour']], 0, 255, -1)
    
    white_region = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=paper_mask)
    white_region_rgb = cv2.cvtColor(white_region, cv2.COLOR_BGR2RGB)
    
    # 背景分割
    combined_mask = cv2.bitwise_or(e_coli_mask, paper_mask)
    background_mask = cv2.bitwise_not(combined_mask)
    background_region = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=background_mask)
    background_region_rgb = cv2.cvtColor(background_region, cv2.COLOR_BGR2RGB)
    
    # 创建分割结果可视化
    segmentation_visual = np.zeros_like(enhanced_rgb)
    segmentation_visual[e_coli_mask > 0] = [255, 255, 0]    # 大肠杆菌 - 黄色
    segmentation_visual[paper_mask > 0] = [255, 255, 255]   # 滤纸片 - 白色
    segmentation_visual[background_mask > 0] = [128, 128, 128]  # 背景 - 灰色
    
    print(f"区域分割完成:")
    print(f"  大肠杆菌区域: {np.count_nonzero(e_coli_mask)} 像素")
    print(f"  滤纸片数量: {len(paper_disks)} 个")
    
    return {
        'e_coli_mask': e_coli_mask,
        'white_mask': white_mask,
        'paper_mask': paper_mask,
        'background_mask': background_mask,
        'e_coli_region': e_coli_region_rgb,
        'white_region': white_region_rgb,
        'background_region': background_region_rgb,
        'segmentation_visual': segmentation_visual,
        'enhanced_bgr': enhanced_bgr,
        'enhanced_rgb': enhanced_rgb,
        'paper_disks': paper_disks
    }

def detect_inhibition_zone_for_disk(disk, enhanced_bgr, e_coli_mask):
    """
    为单个滤纸片检测抑菌圈（增强版验证）
    """
    paper_center = disk['center']
    paper_radius = disk['radius']
    
    # 转换为灰度图进行边缘检测
    gray_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    
    # 使用Canny边缘检测
    edges = cv2.Canny(gray_image, 50, 150)
    
    # 从滤纸片中心向外辐射扫描寻找边界点
    boundary_points = []
    
    # 扫描角度从0到360度，每5度扫描一次
    for angle in range(0, 360, 5):
        # 计算射线方向
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        
        # 从滤纸片边缘开始向外扫描
        current_radius = paper_radius + 10
        max_radius = min(enhanced_bgr.shape[1] - paper_center[0], 
                        enhanced_bgr.shape[0] - paper_center[1],
                        paper_center[0], paper_center[1]) - 10
        
        boundary_found = False
        last_e_coli_value = 0
        
        while current_radius < max_radius and not boundary_found:
            x = int(paper_center[0] + current_radius * dx)
            y = int(paper_center[1] + current_radius * dy)
            
            if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                if edges[y, x] > 0:
                    boundary_points.append([x, y])
                    boundary_found = True
                    break
                
                current_e_coli_value = e_coli_mask[y, x]
                if current_radius > paper_radius + 20:
                    if last_e_coli_value == 0 and current_e_coli_value > 0:
                        boundary_points.append([x, y])
                        boundary_found = True
                        break
                
                last_e_coli_value = current_e_coli_value
            
            current_radius += 1
    
    # 保存原始点数和标准差用于验证
    original_point_count = len(boundary_points)
    original_std = 0
    if boundary_points:
        distances = [np.sqrt((p[0]-paper_center[0])**2 + (p[1]-paper_center[1])**2) for p in boundary_points]
        original_std = np.std(distances)
    
    # 增强版统计滤波
    if len(boundary_points) >= 5:
        filtered_points, angle_coverage, filtered_std = statistical_filter_enhanced(boundary_points, paper_center, paper_radius)
        boundary_points = filtered_points
    else:
        angle_coverage = 0
        filtered_std = original_std
    
    # 使用圆形拟合确定抑菌圈边界
    inhibition_info = {
        'boundary_points': np.array(boundary_points),
        'inhibition_center': None,
        'inhibition_radius': 0,
        'inhibition_diameter': 0,
        'valid': False,
        'original_points': original_point_count,
        'filtered_points': len(boundary_points),  # 修正键名
        'angle_coverage': angle_coverage,
        'fitting_error': 0
    }
    
    if len(boundary_points) >= 5:
        boundary_points_array = np.array(boundary_points)
        (inhibition_center_x, inhibition_center_y), inhibition_radius = cv2.minEnclosingCircle(boundary_points_array)
        inhibition_center = (int(inhibition_center_x), int(inhibition_center_y))
        inhibition_radius = int(inhibition_radius)
        inhibition_diameter = 2 * inhibition_radius
        
        # 计算拟合误差
        fitting_error = calculate_fitting_error(boundary_points, inhibition_center, inhibition_radius)
        inhibition_info['fitting_error'] = fitting_error
        
        # 增强版验证
        is_valid = validate_inhibition_zone(
            boundary_points, inhibition_center, inhibition_radius, 
            paper_radius, angle_coverage, original_std
        )
        
        if is_valid:
            inhibition_info.update({
                'inhibition_center': inhibition_center,
                'inhibition_radius': inhibition_radius,
                'inhibition_diameter': inhibition_diameter,
                'valid': True
            })
            print(f"    ✓ 抑菌圈验证通过")
        else:
            print(f"    ✗ 抑菌圈验证失败")
    
    return inhibition_info

def statistical_filter_enhanced(boundary_points, paper_center, paper_radius):
    """
    增强版统计滤波 - 更严格的异常点过滤
    """
    if len(boundary_points) < 5:
        return boundary_points, 0, 0  # 点数太少，不进行过滤
    
    # 计算每个边界点到滤纸片中心的距离
    distances = []
    for point in boundary_points:
        dist = np.sqrt((point[0] - paper_center[0])**2 + (point[1] - paper_center[1])**2)
        distances.append(dist)
    
    # 第一轮滤波：使用严格的标准差倍数（±1.5倍）
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    
    print(f"  统计滤波前: {len(boundary_points)} 个点")
    print(f"  距离统计: 中位数={median_dist:.1f}px, 标准差={std_dist:.1f}px")
    
    # 严格过滤阈值（中位数±1.5倍标准差）
    lower_bound = median_dist - 1.5 * std_dist
    upper_bound = median_dist + 1.5 * std_dist
    
    # 确保阈值在合理范围内
    lower_bound = max(lower_bound, paper_radius * 1.2)   # 至少比滤纸片大20%
    upper_bound = min(upper_bound, paper_radius * 8)     # 最大不超过滤纸片8倍
    
    print(f"  严格过滤范围: {lower_bound:.1f}px - {upper_bound:.1f}px")
    
    # 第一轮过滤
    filtered_points = []
    filtered_distances = []
    
    for i, point in enumerate(boundary_points):
        dist = distances[i]
        if lower_bound <= dist <= upper_bound:
            filtered_points.append(point)
            filtered_distances.append(dist)
    
    # 如果第一轮过滤后还有足够点数，进行第二轮迭代滤波
    if len(filtered_points) >= 10:
        second_median = np.median(filtered_distances)
        second_std = np.std(filtered_distances)
        
        # 第二轮更严格的过滤（±1.2倍标准差）
        second_lower = second_median - 1.2 * second_std
        second_upper = second_median + 1.2 * second_std
        
        second_lower = max(second_lower, paper_radius * 1.2)
        second_upper = min(second_upper, paper_radius * 8)
        
        final_points = []
        final_distances = []
        
        for i, point in enumerate(filtered_points):
            dist = filtered_distances[i]
            if second_lower <= dist <= second_upper:
                final_points.append(point)
                final_distances.append(dist)
        
        filtered_points = final_points
        filtered_distances = final_distances
        print(f"  第二轮迭代滤波后: {len(filtered_points)} 个点")
    
    # 计算边界点角度分布
    angle_coverage = calculate_angle_coverage(filtered_points, paper_center)
    
    return filtered_points, angle_coverage, std_dist

def calculate_angle_coverage(points, center):
    """
    计算边界点在圆周上的角度覆盖范围
    """
    if len(points) < 3:
        return 0
    
    angles = []
    for point in points:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 360  # 转换为0-360度
        angles.append(angle)
    
    # 排序角度
    angles.sort()
    
    # 计算最大角度间隙
    max_gap = 0
    for i in range(len(angles)):
        gap = (angles[(i + 1) % len(angles)] - angles[i]) % 360
        if gap > max_gap:
            max_gap = gap
    
    # 角度覆盖范围 = 360度 - 最大间隙
    coverage = 360 - max_gap
    return coverage

def validate_inhibition_zone(boundary_points, inhibition_center, inhibition_radius, paper_radius, angle_coverage, original_std):
    """
    验证抑菌圈的合理性
    """
    # 1. 最小点数检查
    if len(boundary_points) < 15:  # 至少需要15个点
        print(f"    点数不足: {len(boundary_points)} < 15")
        return False
    
    # 2. 角度分布检查
    if angle_coverage < 180:  # 至少覆盖180度圆周
        print(f"    角度覆盖不足: {angle_coverage:.1f}° < 180°")
        return False
    
    # 3. 半径合理性检查
    if inhibition_radius < paper_radius * 1.3:  # 至少比滤纸片大30%
        print(f"    抑菌圈太小: {inhibition_radius:.1f} < {paper_radius * 1.3:.1f}")
        return False
    
    if inhibition_radius > paper_radius * 6:    # 最大不超过滤纸片6倍
        print(f"    抑菌圈太大: {inhibition_radius:.1f} > {paper_radius * 6:.1f}")
        return False
    
    # 4. 边界点分散度检查（针对无抑菌圈情况）
    if original_std > paper_radius * 1.5:  # 如果原始边界点过于分散
        print(f"    边界点过于分散: 标准差{original_std:.1f} > {paper_radius * 1.5:.1f}")
        return False
    
    # 5. 拟合质量检查
    if len(boundary_points) >= 5:
        avg_error = calculate_fitting_error(boundary_points, inhibition_center, inhibition_radius)
        if avg_error > inhibition_radius * 0.3:  # 平均误差不超过半径的30%
            print(f"    拟合质量差: 平均误差{avg_error:.1f} > {inhibition_radius * 0.3:.1f}")
            return False
    
    # 6. 边界点密度检查
    point_density = len(boundary_points) / 72  # 72个扫描角度
    if point_density < 0.3:  # 至少30%的扫描角度找到边界点
        print(f"    边界点密度低: {point_density:.2f} < 0.3")
        return False
    
    return True

def calculate_fitting_error(points, center, radius):
    """
    计算边界点到拟合圆的平均距离误差
    """
    errors = []
    for point in points:
        dist_to_center = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        error = abs(dist_to_center - radius)
        errors.append(error)
    
    return np.mean(errors) if errors else float('inf')

def statistical_filter(boundary_points, paper_center, paper_radius):
    """
    基于距离中位数和标准差的统计滤波
    剔除偏离主要群体的异常边界点
    """
    if len(boundary_points) < 5:
        return boundary_points  # 点数太少，不进行过滤
    
    # 计算每个边界点到滤纸片中心的距离
    distances = []
    for point in boundary_points:
        dist = np.sqrt((point[0] - paper_center[0])**2 + (point[1] - paper_center[1])**2)
        distances.append(dist)
    
    # 计算距离的中位数和标准差
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    
    print(f"  统计滤波前: {len(boundary_points)} 个点")
    print(f"  距离统计: 中位数={median_dist:.1f}px, 标准差={std_dist:.1f}px")
    
    # 设置过滤阈值（中位数±2倍标准差）
    lower_bound = median_dist - 2 * std_dist
    upper_bound = median_dist + 2 * std_dist
    
    # 确保阈值在合理范围内（至少大于滤纸片半径）
    lower_bound = max(lower_bound, paper_radius * 1.1)  # 至少比滤纸片大10%
    upper_bound = min(upper_bound, paper_radius * 15)   # 最大不超过滤纸片15倍
    
    print(f"  过滤范围: {lower_bound:.1f}px - {upper_bound:.1f}px")
    
    # 过滤异常点
    filtered_points = []
    filtered_distances = []
    
    for i, point in enumerate(boundary_points):
        dist = distances[i]
        if lower_bound <= dist <= upper_bound:
            filtered_points.append(point)
            filtered_distances.append(dist)
    
    # 如果过滤后点数太少，放宽条件重新过滤
    if len(filtered_points) < 5 and len(boundary_points) >= 5:
        print("  过滤后点数过少，使用宽松条件重新过滤")
        # 使用中位数±3倍标准差
        lower_bound = max(median_dist - 3 * std_dist, paper_radius * 1.1)
        upper_bound = min(median_dist + 3 * std_dist, paper_radius * 15)
        
        filtered_points = []
        filtered_distances = []
        for i, point in enumerate(boundary_points):
            dist = distances[i]
            if lower_bound <= dist <= upper_bound:
                filtered_points.append(point)
                filtered_distances.append(dist)
    
    # 最终检查：确保仍有足够点数
    if len(filtered_points) < 3:
        print("  过滤后点数不足，返回原始点集")
        return boundary_points
    
    # 计算过滤后的统计信息
    if filtered_distances:
        filtered_median = np.median(filtered_distances)
        filtered_std = np.std(filtered_distances)
        print(f"  统计滤波后: {len(filtered_points)} 个点")
        print(f"  过滤后距离统计: 中位数={filtered_median:.1f}px, 标准差={filtered_std:.1f}px")
    
    return filtered_points

def detect_inhibition_zones(segmented_data):
    """
    为所有滤纸片检测抑菌圈
    """
    print("开始抑菌圈边界识别...")
    
    paper_disks = segmented_data['paper_disks']
    enhanced_bgr = segmented_data['enhanced_bgr']
    enhanced_rgb = segmented_data['enhanced_rgb']
    e_coli_mask = segmented_data['e_coli_mask']
    
    inhibition_zones = []
    
    # 为每个滤纸片检测抑菌圈
    for i, disk in enumerate(paper_disks):
        print(f"检测滤纸片 {i+1} 的抑菌圈...")
        inhibition_info = detect_inhibition_zone_for_disk(disk, enhanced_bgr, e_coli_mask)
        
        # 计算实际尺寸（滤纸片直径6mm）
        paper_diameter_px = disk['diameter']
        pixel_to_mm_ratio = 6.0 / paper_diameter_px  # 6mm / 像素直径
        
        if inhibition_info['valid']:
            # 计算抑菌圈实际直径（mm）
            inhibition_diameter_mm = inhibition_info['inhibition_diameter'] * pixel_to_mm_ratio
            
            print(f"  滤纸片 {i+1}: 找到抑菌圈")
            print(f"    滤纸片直径: {paper_diameter_px} px = 6.0 mm")
            print(f"    抑菌圈直径: {inhibition_info['inhibition_diameter']} px = {inhibition_diameter_mm:.2f} mm")
            print(f"    边界点: {inhibition_info['filtered_points']} 个有效点 (原始{inhibition_info['original_points']}个)")
            
            # 添加到结果中
            inhibition_info['inhibition_diameter_mm'] = inhibition_diameter_mm
            inhibition_info['pixel_to_mm_ratio'] = pixel_to_mm_ratio
        else:
            print(f"  滤纸片 {i+1}: 未找到有效的抑菌圈")
            print(f"    滤纸片直径: {paper_diameter_px} px = 6.0 mm")
            print(f"    边界点: {inhibition_info['filtered_points']} 个有效点 (原始{inhibition_info['original_points']}个)")
        
        inhibition_zones.append({
            'paper_disk': disk,
            'inhibition_info': inhibition_info
        })
    
    # 创建可视化结果
    result_visual = enhanced_rgb.copy()
    
    # 为每个滤纸片和抑菌圈绘制
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, zone in enumerate(inhibition_zones):
        disk = zone['paper_disk']
        inhibition_info = zone['inhibition_info']
        color = colors[i % len(colors)]
        
        # 标记滤纸片
        cv2.circle(result_visual, disk['center'], 5, color, -1)
        cv2.circle(result_visual, disk['center'], disk['radius'], color, 2)
        cv2.putText(result_visual, f"Paper{i+1}", 
                   (disk['center'][0] + disk['radius'] + 5, disk['center'][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 标记抑菌圈（如果存在）
        if inhibition_info['valid']:
            cv2.circle(result_visual, inhibition_info['inhibition_center'], 5, color, -1)
            cv2.circle(result_visual, inhibition_info['inhibition_center'], 
                      inhibition_info['inhibition_radius'], color, 2)
            
            # 显示实际直径
            diameter_mm = inhibition_info['inhibition_diameter_mm']
            cv2.putText(result_visual, f"Inhibition{i+1}: {diameter_mm:.1f}mm", 
                       (inhibition_info['inhibition_center'][0] + inhibition_info['inhibition_radius'] + 5, 
                        inhibition_info['inhibition_center'][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 标记边界点
        for point in inhibition_info['boundary_points']:
            cv2.circle(result_visual, tuple(point), 2, color, -1)
    
    # 添加统计信息
    valid_zones = [z for z in inhibition_zones if z['inhibition_info']['valid']]
    cv2.putText(result_visual, f"Found {len(valid_zones)} inhibition zones", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    segmented_data.update({
        'inhibition_zones': inhibition_zones,
        'result_visual': result_visual
    })
    
    return segmented_data

def display_final_result_only(processed_data, segmented_data):
    """
    只显示最终检测结果 - 只输出前四个抑菌圈
    """
    plt.figure(figsize=(15, 10), dpi=100)
    
    # 左侧显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(processed_data['original'])
    plt.title('Original Image')
    plt.axis('off')
    
    # 右侧显示最终检测结果
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_data.get('result_visual', segmented_data['enhanced_rgb']))
    plt.title('Inhibition Zone Detection Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 在控制台输出简洁的检测结果 - 只输出前四个
    if 'inhibition_zones' in segmented_data:
        # 只取前四个滤纸片
        first_four_zones = segmented_data['inhibition_zones'][:4]
        valid_zones = [z for z in first_four_zones if z['inhibition_info']['valid']]
        
        print(f"\n=== 抑菌圈检测最终结果 (前四个滤纸片) ===")
        print(f"检测到 {len(valid_zones)} 个有效的抑菌圈")
        
        for i, zone in enumerate(first_four_zones):
            disk = zone['paper_disk']
            inhibition_info = zone['inhibition_info']
            
            if inhibition_info['valid']:
                print(f"\n滤纸片 {i+1}:")
                print(f"  • 位置: {disk['center']}")
                print(f"  • 滤纸片直径: {disk['diameter']} px = 6.0 mm")
                print(f"  • 抑菌圈直径: {inhibition_info['inhibition_diameter']} px = {inhibition_info['inhibition_diameter_mm']:.2f} mm")
                # print(f"  • 抑菌效果: {inhibition_info['inhibition_radius']/disk['radius']:.2f} 倍")
            else:
                print(f"\n滤纸片 {i+1}: 未检测到明显的抑菌圈")
                print(f"  • 滤纸片直径: {disk['diameter']} px = 6.0 mm")

def display_detailed_results(processed_data, segmented_data):
    """
    显示详细处理过程 - 只输出前四个抑菌圈
    """
    # 设置图形大小
    plt.figure(figsize=(20, 15), dpi=100)
    
    # 第一行：预处理和分割结果
    plt.subplot(3, 4, 1)
    plt.imshow(processed_data['original'])
    plt.title('1. Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(processed_data['enhanced'])
    plt.title('2. Enhanced Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(segmented_data['e_coli_region'])
    plt.title('3. E.coli Region')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(segmented_data['white_region'])
    plt.title('4. Filter Papers')
    plt.axis('off')
    
    # 第二行：掩码和分割可视化
    plt.subplot(3, 4, 5)
    plt.imshow(segmented_data['e_coli_mask'], cmap='gray')
    plt.title('5. E.coli Mask')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(segmented_data['white_mask'], cmap='gray')
    plt.title('6. All White Regions')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(segmented_data['paper_mask'], cmap='gray')
    plt.title('7. Filtered Papers')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(segmented_data['segmentation_visual'])
    plt.title('8. Segmentation Result')
    plt.axis('off')
    
    # 第三行：抑菌圈检测结果
    plt.subplot(3, 4, 9)
    # 显示边缘检测结果
    gray_image = cv2.cvtColor(segmented_data['enhanced_bgr'], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    plt.imshow(edges, cmap='gray')
    
    # 标记边界点 - 只标记前四个
    if 'inhibition_zones' in segmented_data:
        first_four_zones = segmented_data['inhibition_zones'][:4]
        for zone in first_four_zones:
            points = zone['inhibition_info']['boundary_points']
            if len(points) > 0:
                plt.scatter(points[:, 0], points[:, 1], c='red', s=10, alpha=0.6)
    plt.title('9. Boundary Points on Edges (First 4)')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(segmented_data.get('result_visual', segmented_data['enhanced_rgb']))
    plt.title('10. Final Detection Result')
    plt.axis('off')
    
    # 显示检测结果统计 - 只显示前四个
    plt.subplot(3, 4, 11)
    plt.axis('off')
    
    # 创建文本信息 - 只显示前四个
    if 'inhibition_zones' in segmented_data:
        first_four_zones = segmented_data['inhibition_zones'][:4]
        info_text = "Detection Results (First 4):\n\n"
        valid_count = 0
        for i, zone in enumerate(first_four_zones):
            disk = zone['paper_disk']
            inhibition_info = zone['inhibition_info']
            
            info_text += f"Paper {i+1}:\n"
            info_text += f"  Center: {disk['center']}\n"
            info_text += f"  Radius: {disk['radius']}px\n"
            
            if inhibition_info['valid']:
                valid_count += 1
                info_text += f"  Inhibition Radius: {inhibition_info['inhibition_radius']}px\n"
                info_text += f"  Inhibition Diameter: {inhibition_info['inhibition_diameter']}px\n"
                info_text += f"  Inhibition Diameter: {inhibition_info['inhibition_diameter_mm']:.2f}mm\n"
            else:
                info_text += f"  No valid inhibition zone\n"
            info_text += "\n"
        
        info_text += f"Total valid: {valid_count}/4"
        
        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.subplot(3, 4, 12)
    plt.axis('off')
    plt.title('12. Analysis Complete')
    
    plt.tight_layout()
    plt.show()
    
    # 控制台输出详细结果 - 只输出前四个
    if 'inhibition_zones' in segmented_data:
        first_four_zones = segmented_data['inhibition_zones'][:4]
        valid_zones = [z for z in first_four_zones if z['inhibition_info']['valid']]
        
        print(f"\n=== 抑菌圈检测结果汇总 (前四个滤纸片) ===")
        print(f"找到 {len(valid_zones)} 个有效的抑菌圈")
        
        for i, zone in enumerate(first_four_zones):
            disk = zone['paper_disk']
            inhibition_info = zone['inhibition_info']
            
            print(f"\n滤纸片 {i+1}:")
            print(f"  中心位置: {disk['center']}")
            print(f"  • 滤纸片直径: {disk['diameter']} px = 6.0 mm")
            # print(f"  滤纸片面积: {disk['area']:.1f} 像素²")
            # print(f"  圆形度: {disk['circularity']:.3f}")
            
            if inhibition_info['valid']:
                # print(f"  抑菌圈半径: {inhibition_info['inhibition_radius']} 像素")
                print(f"  • 抑菌圈直径: {inhibition_info['inhibition_diameter']} px = {inhibition_info['inhibition_diameter_mm']:.2f} mm")
                # print(f"  抑菌圈/滤纸片半径比: {inhibition_info['inhibition_radius']/disk['radius']:.2f}")
            else:
                print(f"  未检测到有效的抑菌圈")
                # print(f"  找到的边界点数量: {len(inhibition_info['boundary_points'])}")

def main():
    """
    前提是每个照片都是像例子一样的黄色和光照条件
    针对白色和黄色区域识别分割
    识别滤纸片边缘
    边缘检测抑菌圈
    感谢北の猫给我找了这么个好玩的项目
    """
    
    print("抑菌圈自动检测系统")
    print("请保证大肠杆菌的颜色没有太大变化和光照条件相同")
    
    # 设置中文字体
    setup_chinese_font()
    
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='抑菌圈自动检测系统')
    parser.add_argument('-input', '--input', type=str, required=True, 
                       help='输入图片路径')
    parser.add_argument('-mode', '--mode', type=str, choices=['simple', 'detailed'], 
                       default='simple', help='显示模式: simple(简单) 或 detailed(详细)')
    
    # 解析参数
    args = parser.parse_args()
    image_path = args.input
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return
    
    
    try:
        print(f"开始处理图片: {image_path}")
        print("Starting image preprocessing...")
        # 图像预处理
        processed_data = preprocess_image(image_path)
        print("Image preprocessing completed!")
        
        print("Starting region segmentation...")
        # 区域分割
        segmented_data = region_segmentation(processed_data)
        print("Region segmentation completed!")
        
        print("Starting inhibition zone detection...")
        # 抑菌圈边界识别
        segmented_data = detect_inhibition_zones(segmented_data)
        print("Inhibition zone detection completed!")
        
        # 根据参数选择显示模式
        if args.mode == "detailed":
            print("Displaying detailed results...")
            display_detailed_results(processed_data, segmented_data)
        else:
            print("Displaying final result only...")
            display_final_result_only(processed_data, segmented_data)
        

        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()



# 在文件末尾添加
if __name__ == "__main__":
    main()
