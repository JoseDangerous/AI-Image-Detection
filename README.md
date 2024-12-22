# Example Submission: Investigating and Detecting AI-Generated Images

---

## Step 1: Media Type Selection

**Chosen Media Type:** Images  
**Rationale:** I selected images because the visual nature of AI-generated media often presents identifiable patterns or artifacts that distinguish it from human-generated images. Exploring these features will provide insights into how AI-generated images differ from real ones.

---

## Step 2: Collected Examples

- **Images 1–3:** AI-generated images collected from the internet.  
- **Images 4–6:** Recent pictures of my three kids.

---

## Step 3: Brainstorm Specific Traits with AI

**Prompt Used:**  
*"What traits can be analyzed to determine if an image is AI-generated? Consider artifacts, inconsistencies, or other distinguishing characteristics."*

**AI Response:**
1. Symmetry Issues: AI-generated faces often exhibit unnatural symmetry or asymmetry.
2. Edge Artifacts: AI images may have blurry or jagged edges in objects or facial features.
3. Lighting and Shadows: Lighting in AI images may be inconsistent or unrealistic.
4. Background Artifacts: AI-generated images may include nonsensical or incomplete details in the background.
5. Color Variance: AI images sometimes display unnatural color gradients or inconsistencies.

**Traits Selected for Analysis:**
1. Symmetry Issues  
2. Eye Alignment  
3. Texture Artifacts  

**Rationale:** These traits are visually observable and align with known limitations of AI-generated images. They also provide measurable criteria for programmatic analysis. These traits were selected after the initial traits (symmetry, edge artifacts, and background artifacts) failed to differentiate between AI and human images even when the differences were fairly obvious.

---

## Step 5: Program Testing

| **Image** | **Prediction**     | **Actual**          | **Correct?** |
|-----------|--------------------|---------------------|--------------|
| Image 1   | AI-generated       | AI-generated        | Yes          |
| Image 2   | AI-generated       | AI-generated        | Yes          |
| Image 3   | AI-generated       | AI-generated        | Yes          |
| Image 4   | Real human face    | Real human face     | Yes          |
| Image 5   | Real human face    | Real human face     | Yes          |
| Image 6   | Real human face    | Real human face     | Yes          |

**Result:** The program correctly identified all six images. The detection based on symmetry, eye alignment, and texture analysis worked effectively. Texture analysis seemed to be the feature that best distinguished between AI-generated images and real human faces.

---

## Step 6: Reflection Report

### 1. Program Performance:
- The program was able to distinguish between human and AI-generated faces.  
- It required significant trial and error and back-and-forth messaging with the AI to reach this point.  
- Initially, the program struggled with unnatural AI images, labeling all images as either AI or human.  
- Fine-tuning using symmetry issues, eye alignment, and texture artifacts improved accuracy.  
- Adjusting thresholds for unnatural symmetry and emphasizing the texture score further enhanced the model's performance.

### 2. Feature Analysis:
- **Symmetry Issues:** Faces with greater symmetry were more likely AI-generated. However, most faces—human and AI—were not very symmetric, making this a less critical feature.
- **Eye Alignment:** Misaligned eyes were a potential indicator of AI generation but were not significant in the tested images.
- **Texture Artifacts:** The most important feature for detecting AI images, particularly for distinguishing realistic AI-generated images from human ones.

### 3. Limitations and Improvements:
- **Limitations:** The program was tested on only six images, which limits its applicability, especially for high-quality AI images.  
- **Improvements:** Adding features like color variance or analyzing reflections and shadows could improve detection. Expanding the dataset could enhance accuracy for subtle differences.

---

## Conclusion

This project deepened my understanding of AI-generated images and detection methods. I was surprised by the importance of texture artifacts in detecting AI images and the program's ability to identify two fairly natural-looking AI images. While the program performed well on this small dataset, future improvements could address its limitations in detecting more sophisticated AI-generated images.

---
