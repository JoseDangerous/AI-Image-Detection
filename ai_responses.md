## Model Used: GitHub Copilot using Claude 3.7 Sonnet (Preview)

**Prompt Used:**
*What traits and characteristics can be analyzed to determine if an image is AI-generated? Consider artifacts, inconsistencies, or other distinguishing characteristics.*

**AI Response:**
Here's a comprehensive response to your question about traits for detecting AI-generated images:

# Traits for Detecting AI-Generated Images

1. **Facial Anomalies**
   - Asymmetrical facial features
   - Unnatural eye positioning or shapes
   - Irregular teeth patterns
   - Distorted or blended ears
   - Unrealistic hair textures and boundaries

2. **Textural Inconsistencies**
   - Overly smooth or plastic-like skin textures
   - Repetitive patterns in backgrounds
   - Unnatural fabric folds or textures
   - Inconsistent surface reflections

3. **Structural Impossibilities**
   - Anatomically impossible body proportions
   - Physics-defying object arrangements
   - Impossible architectural elements
   - Incorrect shadow casting

4. **Background Oddities**
   - Blurry or warped backgrounds
   - Objects blending into backgrounds unnaturally
   - Inconsistent perspective or depth
   - Illogical scene composition

5. **Technical Artifacts**
   - Edge bleeding or haloing effects
   - Inconsistent image resolution across elements
   - Unnatural color transitions
   - Signature distortions in specific elements (hands, fingers, text)

6. **Metadata Analysis**
   - Absence of camera metadata (EXIF data)
   - Unusual compression artifacts
   - Inconsistent noise patterns
   - Model-specific generation patterns

7. **Lighting and Reflections**
   - Inconsistent light sources
   - Physically impossible reflections
   - Mismatch between shadows and light direction
   - Unrealistic highlights on materials

8. **Fine Detail Analysis**
   - Nonsensical text or symbols
   - Irregular object boundaries
   - Missing or duplicated small details
   - Watermark distortions or remnants

These characteristics can be used as features for machine learning algorithms to detect AI-generated imagery with increasing accuracy.
---

**Prompt Used:**
*Please create a Python program to determine if an image is AI-generated or human-generated. The program should analyze the following traits: textural inconsistencies, technical artifacts, and fine detail analysis. Use image processing libraries like OpenCV or PIL. Please include detailed comments explaining each step of the code. Additionally, explain how to install any needed dependencies.*

**AI Response:**
# AI Image Detection Program

Below is a Python program that analyzes images to detect whether they're AI-generated or human-created, focusing on the traits you identified: textural inconsistencies, technical artifacts, and fine detail analysis.

## Program Implementation

```python
import cv2
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path

class AIImageDetector:
    def __init__(self):
        """Initialize the AI image detector with detection parameters."""
        # Thresholds for different detection features
        self.noise_threshold = 0.04    # Threshold for noise variance
        self.edge_threshold = 0.15     # Threshold for edge consistency
        self.texture_threshold = 0.65  # Threshold for texture analysis
        self.detail_threshold = 0.5    # Threshold for fine detail analysis

    def analyze_image(self, image_path):
        """
        Analyze an image to determine if it's AI-generated.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Results including prediction and feature scores
        """
        try:
            # Load image with both OpenCV (for processing) and PIL (for metadata)
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return {"error": "Could not load image"}
                
            img_pil = Image.open(image_path)
            
            # Convert to RGB for consistent processing
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Run individual feature analyses
            texture_score = self._analyze_textures(img_cv)
            artifact_score = self._analyze_technical_artifacts(img_cv)
            detail_score = self._analyze_fine_details(img_cv)
            metadata_score = self._analyze_metadata(img_pil)
            
            # Calculate final score (weighted average)
            final_score = (texture_score * 0.35 + 
                          artifact_score * 0.35 + 
                          detail_score * 0.2 + 
                          metadata_score * 0.1)
            
            # Determine if the image is AI-generated based on composite score
            # Lower scores indicate more likely AI-generated
            is_ai_generated = final_score < 0.5
            
            return {
                "prediction": "AI-Generated" if is_ai_generated else "Human-Generated",
                "confidence": abs(0.5 - final_score) * 2,  # Scale to 0-1 confidence
                "feature_scores": {
                    "texture_analysis": texture_score,
                    "technical_artifacts": artifact_score,
                    "fine_details": detail_score,
                    "metadata": metadata_score
                }
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _analyze_textures(self, img):
        """
        Analyze textural inconsistencies in the image.
        Detects overly smooth textures and unnatural patterns.
        
        Returns:
            float: Score between 0-1 (lower = more likely AI-generated)
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Calculate local binary pattern (simplified version)
        # This helps identify texture patterns
        texture_patterns = np.zeros_like(gray)
        h, w = gray.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 0
                code |= (gray[i-1, j] > center) << 1
                code |= (gray[i-1, j+1] > center) << 2
                code |= (gray[i, j+1] > center) << 3
                code |= (gray[i+1, j+1] > center) << 4
                code |= (gray[i+1, j] > center) << 5
                code |= (gray[i+1, j-1] > center) << 6
                code |= (gray[i, j-1] > center) << 7
                texture_patterns[i, j] = code
                
        # 2. Calculate texture variance (to detect overly smooth areas)
        kernel = np.ones((5, 5), np.float32) / 25
        blurred = cv2.filter2D(gray, -1, kernel)
        variance = np.var(np.abs(gray.astype(float) - blurred.astype(float)))
        normalized_variance = min(1.0, variance / 100.0)  # Normalize to 0-1
        
        # 3. Check for repetitive patterns
        pattern_hist = cv2.calcHist([texture_patterns], [0], None, [256], [0, 256])
        pattern_hist = pattern_hist / pattern_hist.sum()  # Normalize
        pattern_entropy = -np.sum(pattern_hist * np.log2(pattern_hist + 1e-10))
        pattern_score = min(1.0, pattern_entropy / 8.0)  # Normalize to 0-1
        
        # Combine scores (higher variance and entropy = more likely human)
        texture_score = (normalized_variance * 0.6 + pattern_score * 0.4)
        return texture_score

    def _analyze_technical_artifacts(self, img):
        """
        Analyze technical artifacts like edge inconsistencies, 
        color transitions, and resolution inconsistencies.
        
        Returns:
            float: Score between 0-1 (lower = more likely AI-generated)
        """
        # 1. Edge analysis
        edges = cv2.Canny(img, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Natural images tend to have more varied edge patterns
        edge_score = min(1.0, edge_density * 5)
        
        # 2. Color transition analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_transitions = cv2.Sobel(hsv[:,:,0], cv2.CV_64F, 1, 1)
        color_score = np.mean(np.abs(hue_transitions)) / 30.0  # Normalize
        color_score = min(1.0, color_score)
        
        # 3. Noise consistency analysis (AI images often have inconsistent noise)
        # Apply wavelet transform to analyze noise patterns
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        noise = img_gray - cv2.GaussianBlur(img_gray, (5, 5), 0)
        noise_var = np.var(noise)
        noise_score = min(1.0, noise_var / 50.0)
        
        # Combine scores
        artifact_score = (edge_score * 0.4 + color_score * 0.3 + noise_score * 0.3)
        return artifact_score

    def _analyze_fine_details(self, img):
        """
        Analyze fine details like object boundaries, 
        small elements consistency, and potential text issues.
        
        Returns:
            float: Score between 0-1 (lower = more likely AI-generated)
        """
        # 1. High-frequency content analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        high_freq_score = np.var(laplacian) / 1000.0  # Normalize
        high_freq_score = min(1.0, high_freq_score)
        
        # 2. Detail consistency check using wavelet transform
        # This is a simplified version - wavelet analysis can be more complex
        blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
        blur2 = cv2.GaussianBlur(gray, (5, 5), 0)
        detail_layer1 = gray.astype(float) - blur1.astype(float)
        detail_layer2 = blur1.astype(float) - blur2.astype(float)
        
        # Calculate correlation between detail layers
        # Human photos typically have more consistent detail across scales
        correlation = np.corrcoef(detail_layer1.flatten(), detail_layer2.flatten())[0,1]
        detail_score = max(0, min(1.0, (correlation + 1) / 2.0))
        
        # Combine scores
        fine_detail_score = (high_freq_score * 0.5 + detail_score * 0.5)
        return fine_detail_score

    def _analyze_metadata(self, img_pil):
        """
        Analyze image metadata for signs of AI generation.
        
        Returns:
            float: Score between 0-1 (lower = more likely AI-generated)
        """
        # Check for EXIF data
        exif_data = img_pil._getexif() if hasattr(img_pil, '_getexif') else {}
        if exif_data and len(exif_data) > 5:  # Human photos often have richer EXIF data
            return 0.8
        
        # Check for standard dimensions often used in AI generation
        width, height = img_pil.size
        common_ai_dims = [(512, 512), (1024, 1024), (512, 768), (768, 512)]
        for w, h in common_ai_dims:
            if (width == w and height == h) or (width/height == w/h):
                return 0.3
                
        return 0.5  # Neutral if inconclusive


def process_images_in_directory(detector, dir_path):
    """
    Process all images in a directory and print results.
    """
    results = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in valid_extensions):
            print(f"\nAnalyzing: {file}")
            result = detector.analyze_image(file_path)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
                
            print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
            print("Feature scores:")
            for feature, score in result['feature_scores'].items():
                print(f"  - {feature}: {score:.2f}")
                
            # Save result for summary
            results.append({
                'file': file,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
    
    return results


def main():
    """Main function to run the AI image detector."""
    parser = argparse.ArgumentParser(description='Detect AI-generated images')
    parser.add_argument('--image', help='Path to a single image')
    parser.add_argument('--dir', help='Directory containing images to analyze')
    args = parser.parse_args()
    
    detector = AIImageDetector()
    
    if args.image:
        if os.path.isfile(args.image):
            result = detector.analyze_image(args.image)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
                print("Feature scores:")
                for feature, score in result['feature_scores'].items():
                    print(f"  - {feature}: {score:.2f}")
        else:
            print(f"File not found: {args.image}")
            
    elif args.dir:
        if os.path.isdir(args.dir):
            results = process_images_in_directory(detector, args.dir)
            
            # Print summary
            print("\n===== SUMMARY =====")
            for result in results:
                print(f"{result['file']}: {result['prediction']} (Confidence: {result['confidence']:.2f})")
        else:
            print(f"Directory not found: {args.dir}")
            
    else:
        # Use current directory if no arguments provided
        print("No path provided. Analyzing images in the current directory.")
        results = process_images_in_directory(detector, ".")


if __name__ == "__main__":
    main()
```

## Installation Instructions

To use this program, you'll need to install the required dependencies:

```bash
pip install numpy opencv-python pillow
```

## How to Use the Program

You can use the program in two ways:

1. **Analyze a single image**:
   ```bash
   python ai_image_detector.py --image path/to/image.jpg
   ```

2. **Analyze all images in a directory**:
   ```bash
   python ai_image_detector.py --dir path/to/directory
   ```

3. **If run without arguments**, it will analyze all images in the current directory.

## How It Works

This program analyzes images using four main detection strategies:

1. **Texture Analysis**: 
   - Examines local binary patterns to identify unnatural texture patterns
   - Calculates texture variance to detect overly smooth areas common in AI-generated images
   - Evaluates entropy of texture patterns to identify repetitive elements

2. **Technical Artifacts Analysis**:
   - Analyzes edge consistency and density
   - Examines color transitions for unnatural patterns
   - Evaluates noise consistency across the image

3. **Fine Detail Analysis**:
   - Measures high-frequency content distribution
   - Compares detail consistency across multiple scales
   - Uses wavelet-inspired approach to examine detail correlation

4. **Metadata Analysis**:
   - Checks for presence of EXIF data (often missing in AI images)
   - Identifies common dimensions used in AI generation

Each analysis returns a score between 0-1, where lower scores indicate traits more commonly found in AI-generated images. The final prediction combines these scores using weighted averaging.

The program outputs the prediction (AI-Generated or Human-Generated), confidence score, and individual feature scores for transparency.
