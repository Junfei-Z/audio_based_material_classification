# audio_based_material_classification
This repo is for CIS 5190(2024 fall) final project which is about the audio classification. (Junfei Zhan)



Version 11/30 - Ensemble Learning Approach

In the November 30th version, I implemented an ensemble learning strategy to enhance classification performance. This ensemble comprises two distinct Logistic Regression models, each utilizing different feature extraction techniques:
	1.	Blackboard Knock Detection Model:
	•	Specialization: Excels at distinguishing audio signals associated with knocking on a blackboard.
	•	Feature Extraction: Tailored to capture characteristics unique to the blackboard knocking sounds.
	2.	General Object Classification Model:
	•	Specialization: Performs effectively in classifying the remaining six object categories.
	•	Feature Extraction: Optimized to identify features relevant to a diverse set of objects.

By using the strengths of both models, the ensemble approach ensures accurate detection of blackboard knocks while maintaining robust classification performance across other object categories. This method capitalizes on the complementary capabilities of each model, resulting in a more reliable and versatile classification system.



For TA, you can just run the python file whose name is run_trained_model.py, whose input and output all are followed the requirement of the leadboard submission!
