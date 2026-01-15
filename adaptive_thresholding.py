import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class AdaptiveThresholdingProcessor:

    def __init__(self):
        self.images = {}  
        self.original_images = {}  
        self.results = {}

    def load_real_images(self):
        print("Loading images from Downloads folder...")

        downloads_path = os.path.expanduser('~/Downloads')

        image_paths = {
            'indoor': os.path.join(downloads_path, 'indoor scene.jpg'),
            'outdoor': os.path.join(downloads_path, 'outdoor scene.webp'),
            'closeup': os.path.join(downloads_path, 'close up.jpg')
        }

        for scene_type, path in image_paths.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Could not load {path}")
                    continue

                self.original_images[scene_type] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.images[scene_type] = gray
                print(f"  Loaded {scene_type} scene: {gray.shape}")
            else:
                print(f"Warning: Image not found at {path}")

        if len(self.images) == 0:
            raise FileNotFoundError("No images could be loaded from Downloads folder")

        print(f"Successfully loaded {len(self.images)} images.")

    def create_synthetic_test_images(self):
        print("Creating synthetic test images...")

        indoor = np.ones((600, 800), dtype=np.uint8) * 200
        cv2.rectangle(indoor, (100, 400), (300, 550), 80, -1)  
        cv2.rectangle(indoor, (500, 300), (700, 500), 120, -1)  
        cv2.circle(indoor, (400, 200), 80, 60, -1)  
        for i in range(600):
            for j in range(800):
                indoor[i, j] = np.clip(indoor[i, j] - j // 10, 0, 255)
        noise = np.random.normal(0, 10, indoor.shape)
        indoor = np.clip(indoor + noise, 0, 255).astype(np.uint8)
        self.images['indoor'] = indoor

        outdoor = np.ones((600, 800), dtype=np.uint8) * 180
        outdoor[0:250, :] = 220
        outdoor[250:600, :] = 100
        cv2.rectangle(outdoor, (100, 150), (150, 400), 40, -1)
        cv2.rectangle(outdoor, (300, 180), (340, 380), 35, -1)
        cv2.rectangle(outdoor, (600, 160), (650, 420), 45, -1)
        cv2.ellipse(outdoor, (200, 100), (80, 40), 0, 0, 360, 240, -1)
        cv2.ellipse(outdoor, (500, 120), (100, 50), 0, 0, 360, 235, -1)
        noise = np.random.normal(0, 8, outdoor.shape)
        outdoor = np.clip(outdoor + noise, 0, 255).astype(np.uint8)
        self.images['outdoor'] = outdoor

        closeup = np.ones((600, 800), dtype=np.uint8) * 190
        cv2.circle(closeup, (400, 300), 180, 100, -1)
        cv2.circle(closeup, (350, 250), 60, 200, -1)
        cv2.ellipse(closeup, (450, 350), (120, 80), 45, 0, 360, 70, -1)
        for i in range(600):
            for j in range(800):
                distance = np.sqrt((i - 200) ** 2 + (j - 300) ** 2)
                closeup[i, j] = np.clip(closeup[i, j] - distance / 8, 0, 255)
        cv2.putText(closeup, 'OBJECT', (330, 310), cv2.FONT_HERSHEY_SIMPLEX,
                   1, 150, 2)
        noise = np.random.normal(0, 12, closeup.shape)
        closeup = np.clip(closeup + noise, 0, 255).astype(np.uint8)
        self.images['closeup'] = closeup

        print("Test images created successfully.")

    def preprocess_image(self, image):
       
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        return denoised

    def otsu_threshold(self, image):
    
        preprocessed = self.preprocess_image(image)

        threshold_value, binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        print(f"  Otsu's threshold value: {threshold_value:.2f}")
        return binary

    def adaptive_mean_threshold(self, image, block_size=15, C=2):
       
        preprocessed = self.preprocess_image(image)

        binary = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, C
        )

        print(f"  Adaptive Mean: block_size={block_size}, C={C}")
        return binary

    def adaptive_gaussian_threshold(self, image, block_size=15, C=2):
       
        preprocessed = self.preprocess_image(image)

        binary = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )

        print(f"  Adaptive Gaussian: block_size={block_size}, C={C}")
        return binary

    def triangle_threshold(self, image):
       
        preprocessed = self.preprocess_image(image)

        threshold_value, binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )

        print(f"  Triangle threshold value: {threshold_value:.2f}")
        return binary

    def process_all_images(self):
        
        methods = {
            'Otsu': self.otsu_threshold,
            'Adaptive Mean': self.adaptive_mean_threshold,
            'Adaptive Gaussian': self.adaptive_gaussian_threshold,
            'Triangle': self.triangle_threshold
        }

        for img_name, image in self.images.items():
            print(f"\nProcessing {img_name} scene:")
            self.results[img_name] = {}

            for method_name, method_func in methods.items():
                print(f"  Applying {method_name}...")
                result = method_func(image)
                self.results[img_name][method_name] = result

    def calculate_segmentation_metrics(self, original, segmented):
        
        total_pixels = segmented.size
        foreground_pixels = np.sum(segmented == 255)
        background_pixels = total_pixels - foreground_pixels

        fg_bg_ratio = foreground_pixels / background_pixels if background_pixels > 0 else 0

        coverage = (foreground_pixels / total_pixels) * 100

        return {
            'foreground_pixels': foreground_pixels,
            'background_pixels': background_pixels,
            'fg_bg_ratio': fg_bg_ratio,
            'coverage_percent': coverage
        }

    def visualize_results(self):
        scene_types = list(self.images.keys())
        methods = ['Otsu', 'Adaptive Mean', 'Adaptive Gaussian', 'Triangle']

        fig, axes = plt.subplots(len(scene_types), len(methods) + 2,
                                figsize=(24, 12))

        for i, scene in enumerate(scene_types):
            axes[i, 0].imshow(self.original_images[scene])
            axes[i, 0].set_title(f'{scene.capitalize()} - Original (Color)', fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(self.images[scene], cmap='gray')
            axes[i, 1].set_title(f'{scene.capitalize()} - Grayscale', fontsize=10)
            axes[i, 1].axis('off')

            for j, method in enumerate(methods):
                axes[i, j + 2].imshow(self.results[scene][method], cmap='gray')
                axes[i, j + 2].set_title(f'{method}',
                                        fontsize=10)
                axes[i, j + 2].axis('off')

        plt.tight_layout()
        plt.savefig('/Users/eli/adaptive_thresholding_results.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'adaptive_thresholding_results.png'")
        plt.close()

    def create_histogram_analysis(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, (scene, image) in enumerate(self.images.items()):
            axes[idx].hist(image.ravel(), bins=256, range=[0, 256],
                          color='blue', alpha=0.7)
            axes[idx].set_title(f'{scene.capitalize()} - Intensity Histogram')
            axes[idx].set_xlabel('Pixel Intensity')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/Users/eli/intensity_histograms.png',
                   dpi=300, bbox_inches='tight')
        print("Histogram analysis saved as 'intensity_histograms.png'")
        plt.close()

    def print_metrics_summary(self):
        print("\n" + "="*80)
        print("SEGMENTATION METRICS SUMMARY")
        print("="*80)

        for scene in self.images.keys():
            print(f"\n{scene.upper()} SCENE:")
            print("-" * 80)

            for method in ['Otsu', 'Adaptive Mean', 'Adaptive Gaussian', 'Triangle']:
                metrics = self.calculate_segmentation_metrics(
                    self.images[scene],
                    self.results[scene][method]
                )

                print(f"\n  {method}:")
                print(f"    Foreground pixels: {metrics['foreground_pixels']:,}")
                print(f"    Background pixels: {metrics['background_pixels']:,}")
                print(f"    FG/BG ratio: {metrics['fg_bg_ratio']:.3f}")
                print(f"    Coverage: {metrics['coverage_percent']:.2f}%")

    def compare_methods(self):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        scene_types = list(self.images.keys())

        for i, scene in enumerate(scene_types):
            diff1 = cv2.absdiff(self.results[scene]['Otsu'],
                               self.results[scene]['Adaptive Mean'])
            axes[i, 0].imshow(diff1, cmap='hot')
            axes[i, 0].set_title(f'{scene} - Otsu vs Adaptive Mean Diff')
            axes[i, 0].axis('off')

            diff2 = cv2.absdiff(self.results[scene]['Otsu'],
                               self.results[scene]['Adaptive Gaussian'])
            axes[i, 1].imshow(diff2, cmap='hot')
            axes[i, 1].set_title(f'{scene} - Otsu vs Adaptive Gaussian Diff')
            axes[i, 1].axis('off')

            diff3 = cv2.absdiff(self.results[scene]['Adaptive Mean'],
                               self.results[scene]['Adaptive Gaussian'])
            axes[i, 2].imshow(diff3, cmap='hot')
            axes[i, 2].set_title(f'{scene} - Mean vs Gaussian Diff')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(self.results[scene]['Triangle'], cmap='gray')
            axes[i, 3].set_title(f'{scene} - Triangle Method')
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig('/Users/eli/method_comparison.png',
                   dpi=300, bbox_inches='tight')
        print("Method comparison saved as 'method_comparison.png'")
        plt.close()


def main():
    print("="*80)
    print("ADAPTIVE THRESHOLDING FOR IMAGE SEGMENTATION")
    print("Module 6 Critical Thinking Option 1")
    print("="*80)

    processor = AdaptiveThresholdingProcessor()

    processor.load_real_images()

    processor.process_all_images()

    print("\nGenerating visualizations...")
    processor.create_histogram_analysis()
    processor.visualize_results()
    processor.compare_methods()

    processor.print_metrics_summary()

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - adaptive_thresholding_results.png")
    print("  - intensity_histograms.png")
    print("  - method_comparison.png")
    print("\nAll adaptive thresholding methods have been applied successfully.")
    print("Review the visualizations to compare the effectiveness of each method.")


if __name__ == "__main__":
    main()
