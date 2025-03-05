# Detecting AI-Generated Images
---

## Step 1: Media Type Selection

**Chosen Media Type:** Images  
**Rationale:**
I chose images because of their ubiquitous use, impact across cultures, and their ability to quickly evoke emotional responses in people.
---

## Step 2: Collected Examples

- **Images 1–3:** AI-generated images collected from the internet.

[AI Image 1 Source](https://thispersondoesnotexist.com/)

[AI Image 2 Link](https://designer.microsoft.com/editor?appliedPrompt=Claymation+style+%5Bboy+with+a+diverse+background%5D+with+a+%5Bhappy+grin%5D%2C+wearing+a+%5Bstriped%2C+button-up+shirt%5D+with+the+name+%5B%22Kian%22%5D+and+%5Bdark+brown%5D+eyes+and+%5Bdark+black+hair+in+a+short+style%5D.+Holding+a+%5Bskateboard%5D+with+%5Bskatepark%5D+in+the+background.+Fashion%2C+text+3D+rendering%2C+typography%2C+illustration%2C+painting%2C+photo%2C+poster%2C+3d+render.+)

[AI Image 3 Link](https://openart.ai/community/O30pSCGraaFKfiKBaJDk)

- **Images 4–6:** Pictures I took and image from a human created movie.
---

## Step 3: Brainstorm Specific Traits with AI

**Prompt Used:**  
*"What traits and characteristics can be analyzed to determine if an image is AI-generated? Consider artifacts, inconsistencies, or other distinguishing characteristics."*

**AI Response: (Check AI Responses document for full responses)**
Here's a comprehensive response to your question about traits for detecting AI-generated images:

**Traits for Detecting AI-Generated Images**

1. Facial Anomalies
2. Textural Inconsistencies
3. Structural Impossibilities
4. Background Oddities
5. Technical Artifacts
6. Metadata Analysis
7. Lighting and Reflections
8. Fine Detail Analysis

These characteristics can be used as features for machine learning algorithms to detect AI-generated imagery with increasing accuracy.

**Traits Selected for Analysis:**

1. **Textural Inconsistencies**
   - Overly smooth or plastic-like skin textures
   - Repetitive patterns in backgrounds
   - Unnatural fabric folds or textures
   - Inconsistent surface reflections

2. **Technical Artifacts**
   - Edge bleeding or haloing effects
   - Inconsistent image resolution across elements
   - Unnatural color transitions
   - Signature distortions in specific elements (hands, fingers, text)

3. **Fine Detail Analysis**
   - Nonsensical text or symbols
   - Irregular object boundaries
   - Missing or duplicated small details
   - Watermark distortions or remnants

**Rationale:** 
These traits tend to be the most difficult for AI to generate correctly or perfectly realistic. They can also become noticeable and easy to spot upon closer inspection.
---

## Step 4: Prompt an AI to Write a Detection Program

**Prompt Used:**
*Please create a Python program to determine if an image is AI-generated or human-generated. The program should analyze the following traits: textural inconsistencies, technical artifacts, and fine detail analysis. Use image processing libraries like OpenCV or PIL. Please include detailed comments explaining each step of the code. Additionally, explain how to install any needed dependencies.*

**AI Response: (Check AI Responses document for full responses)**
---

## Step 5: Program Testing

| **Image** | **Prediction**     | **Actual**          | **Correct?** |
|-----------|--------------------|---------------------|--------------|
| Image 1   | AI-Generated       | AI-Generated        | Yes          |
| Image 2   | Human-Generated    | AI-Generated        | No           |
| Image 3   | AI-Generated       | AI-Generated        | Yes          |
| Image 4   | AI-Generated       | Human-Generated     | No           |
| Image 5   | Human-Generated    | Human-Generated     | Yes          |
| Image 6   | Human-Generated    | Human-Generated     | Yes          |

**Result:** 
The program identified 4 images correctly and 2 incorrectly. The detection based on analysis of texture, technical artifacts, and fine detail seemed to work decently.
---

## Step 6: Reflection Report

### 1. Program Performance:
- The program performed adequately and could identify 4 out of 6 images correctly after only 1 prompt
- One of the examples was an animated 3D character that was classified as Human-Generated even though the background had noticeable artifacts
- The program added another detection strategy, Metadata Analysis, without being prompted to

### 2. Feature Analysis:
- **Textural Inconsistencies:** According to the confidence score given by the program, this was usually the most important feature in determining if an image was AI-Generated. I think this feature stood out the most when looking at the images.

- **Technical Artifacts:** This seemed like the least effective indicator as the images did not display too many artifacts.

- **Fine Detail Analysis:** These details were almost imperceptable to me but seemed to greatly help the program in the determination.

### 3. Limitations and Improvements:
- **Limitations:** The program did not check for all the different features that could've assisted in detecting AI images and only tested 6 images.
- **Improvements:** Testing for more features such as lighting and shadows as well as testing on more images would increase the effectivness of the program.
---

## Conclusion
This project deepened my understanding of AI-generated images, detection methods, and working with AI chat models. It was interesting to see which features would be the most effective and how well the program detected AI images after only a single prompt. While the program worked sufficiently, it can be improved by adding more images to the dataset and using more features.
---