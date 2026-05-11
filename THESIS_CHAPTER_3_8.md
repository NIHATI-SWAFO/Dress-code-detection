### 3. Confusion Matrices

The confusion matrix serves as a vital diagnostic tool to visualize the classification accuracy across all fourteen predefined garment categories. By comparing the model's predictions against the actual ground-truth labels, the matrices highlight the exact distribution of true positive detections versus false classifications.

**Figure X: Normalized Confusion Matrix**  
![Normalized Confusion Matrix](results/val_matrix/confusion_matrix_normalized.png)

**Data Interpretation:**  
The normalized confusion matrix presents the data as percentages, illustrating the proportion of correct versus incorrect predictions for each class. The model demonstrates a strong, distinct diagonal trend, confirming that the vast majority of predictions accurately match the true labels. The most prominent deviation from this diagonal involves the `prohibited_crop_halter` class being misclassified as `civilian_top_short_sleeve`. This specific misclassification is a direct result of noise within the original training dataset, where numerous standard civilian shirts were inaccurately labeled as crop tops. Outside of this isolated issue, misclassifications among the remaining 13 classes are statistically minimal, proving the model's overall reliability in complex garment differentiation.

**Figure X: Absolute Confusion Matrix**  
![Absolute Confusion Matrix](results/val_matrix/confusion_matrix.png)

**Data Interpretation:**  
The absolute confusion matrix provides the raw numerical counts of garments classified during the evaluation phase, which assessed a total of 25,773 bounding boxes. This view confirms the model's stability even when faced with heavily imbalanced classes. For instance, the system successfully distinguishes between 3,608 civilian short-sleeve tops and only 198 uniform tops without exhibiting severe predictive bias toward the majority class.

---

### 4. Evaluation Curves

The evaluation curves visually plot the model's performance metrics across different operational thresholds, offering a comprehensive view of its precision, recall, and optimal confidence settings.

**Figure X: F1-Confidence Curve**  
![F1-Confidence Curve](results/val_matrix/BoxF1_curve.png)

**Data Interpretation:**  
The F1-score represents the harmonic mean of precision and recall, serving as a single metric for the model's overall balance. The F1-Confidence curve demonstrates that the model achieves its peak overall efficiency (a maximum F1-score of 0.86) when the detection confidence threshold is calibrated to 0.306. This indicates that configuring the live deployment system to accept detections with at least a 30.6% confidence score provides the mathematical optimum for catching dress code violations while mitigating the rate of false alarms.

**Figure X: Precision-Recall (PR) Curve**  
![Precision-Recall Curve](results/val_matrix/BoxPR_curve.png)

**Data Interpretation:**  
The Precision-Recall curve plots the model's exactness (Precision) against its completeness (Recall). An optimal detection model produces a curve that approaches the top-right corner of the graph, maximizing the area under the curve. The generated PR curve validates the model's robust mAP@0.50 score of 0.911 (91.1%). This confirms that the system can successfully identify the vast majority of garments without triggering a significant volume of false positive detections.

**Figure X: Precision-Confidence Curve**  
![Precision-Confidence Curve](results/val_matrix/BoxP_curve.png)

**Data Interpretation:**  
This curve illustrates how the model's exactness increases as stricter confidence thresholds are required. The data reveals that when the confidence threshold is elevated to 0.872 (87.2%), every prediction the model makes is 100% correct (Precision = 1.00). This demonstrates that the model possesses high certainty in its strongest bounding box predictions.

**Figure X: Recall-Confidence Curve**  
![Recall-Confidence Curve](results/val_matrix/BoxR_curve.png)

**Data Interpretation:**  
Conversely, the Recall-Confidence curve visualizes the rate at which the model misses actual garments as the required confidence increases. At a baseline confidence threshold of 0.00, the model successfully retrieves 97% of all garments present in the frame, though this leniency would result in an unacceptable volume of false positives in a live deployment. As the confidence threshold rises to the optimal 0.306 mark, recall gracefully stabilizes at approximately 86.1%, confirming a reliable detection rate under operational conditions.
