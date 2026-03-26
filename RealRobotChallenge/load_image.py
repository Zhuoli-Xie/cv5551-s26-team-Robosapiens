import cv2
import sys


def load_image(image_path):
    """
    Load image from file (original, no processing)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    return img


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_image.py <image_path>")
        return

    image_path = sys.argv[1]

    # 读取图片
    img = load_image(image_path)

    # 打印基本信息（方便你确认格式）
    print("Image shape:", img.shape)
    print("Image dtype:", img.dtype)

    # 显示图片
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
