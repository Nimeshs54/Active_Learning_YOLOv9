# Active Learning with YOLOv9

This project introduces an innovative pipeline that integrates Synthetic Data Generation with Active Learning (AL) algorithms to enhance the efficiency of real-world data acquisition and labeling. The pipeline is structured into two distinct phases:

### Phase 1: Synthetic Data Generation
- Create high-fidelity synthetic data samples that closely mimic real-world scenarios.
- Augment the training dataset without extensive manual intervention.
- Establish a strong foundational dataset to improve initial model performance and generalization.

### Phase 2: Active Learning
- Utilize AL techniques to selectively identify and label the most informative real-world data points.
- Optimize labeling efforts and reduce human labor.
- Focus on strategically chosen samples to maximize data acquisition efficiency and minimize associated costs.

---

## 📊 Active Learning Methods Comparison

This comprehensive comparison highlights all three Uncertainty Sampling strategies—**Average**, **Max**, and **Sum**—against **Random Sampling**. It demonstrates:

- **Average and Max** strategies significantly accelerate model learning compared to Random Sampling.
- **Sum** strategy shows its value in extended training scenarios.

![AL Methods Comparison](assets/al_methods_comparison.png)

---

## 📈 All Classes Comparison

This analysis provides a class-specific comparison of three Uncertainty Sampling strategies—**Average**, **Max**, and **Sum**—against **Random Sampling**. It covers performance across eight distinct classes, illustrating how each strategy handles various challenges.

![All Classes Comparison](assets/all_classes_comparison.png)

---

## 🔍 Random Sampling vs. Uncertainty Sampling

Below is a visual comparison of training samples obtained through Random Sampling (RS) and Uncertainty Sampling (US). Training sample sizes range from 50 to 200 samples.

### Random Sampling (Top Row)

<div style="display: flex; justify-content: space-between;">
    <div style="flex: 1; padding: 5px;">
        <img src="assets/rand100.jpg" alt="RS-50" style="width: 100%;"/>
        <p style="text-align: center;">RS-50</p>
    </div>
    <div style="flex: 1; padding: 5px;">
        <img src="assets/rand150.jpg" alt="RS-100" style="width: 100%;"/>
        <p style="text-align: center;">RS-100</p>
    </div>
    <div style="flex: 1; padding: 5px;">
        <img src="assets/rand200.jpg" alt="RS-150" style="width: 100%;"/>
        <p style="text-align: center;">RS-150</p>
    </div>
    <div style="flex: 1; padding: 5px;">
        <img src="assets/rand250.jpg" alt="RS-200" style="width: 100%;"/>
        <p style="text-align: center;">RS-200</p>
    </div>
</div>

<br/>

### Uncertainty Sampling (Bottom Row)

<div style="display: flex; justify-content: space-between;">
    <div style="flex: 1; padding: 5px;">
        <img src="assets/avg100.jpg" alt="US-50" style="width: 100%;"/>
        <p style="text-align: center;">US-50</p>
    </div>
    <div style="flex: 1; padding: 5px;">
        <img src="assets/avg150.jpg" alt="US-100" style="width: 100%;"/>
        <p style="text-align: center;">US-100</p>
    </div>
    <div style="flex: 1; padding: 5px;">
        <img src="assets/avg200.jpg" alt="US-150" style="width: 100%;"/>
        <p style="text-align: center;">US-150</p>
    </div>
    <div style="flex: 1; padding: 5px;">
        <img src="assets/avg250.jpg" alt="US-200" style="width: 100%;"/>
        <p style="text-align: center;">US-200</p>
    </div>
</div>
