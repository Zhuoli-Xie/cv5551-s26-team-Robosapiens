import cv2
import numpy as np

def test_color_mask(image_path):
    # 1. 读取你上传的图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"找不到图片 {image_path}，请检查路径！")
        return

    # 缩放一下图片以便在屏幕上显示 (ZED原图分辨率可能很大)
    scale_percent = 50 # 缩小到 50%
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # 2. 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

    # 3. 应用我们在 Checkpoint 6 中写的红色 HSV 阈值
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 4. 形态学去噪 (开运算和闭运算)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. 将掩膜(Mask)和原图叠加，看看抠出了什么
    # 把 mask 转换成 3 通道才能和彩色原图叠加
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
    
    # 用 bitwise_and 把红方块的彩色部分提取出来，其他地方变黑
    extracted_cube = cv2.bitwise_and(resized_img, mask_3channel)

    # 6. 显示结果
    cv2.imshow("1. Original Image (ZED Camera)", resized_img)
    cv2.imshow("2. Black & White Mask (What the code sees)", mask)
    cv2.imshow("3. Extracted Result", extracted_cube)
    
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替换为你图片的实际名字
    test_color_mask("color.jpg")
