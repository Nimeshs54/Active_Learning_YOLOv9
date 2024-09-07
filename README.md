# Active_Learning_YOLOv9

This work introduces an innovative pipeline that effectively integrates Synthetic Data Generation with Active Learning (AL) algorithms to enhance the efficiency of real-world data acquisition and labeling. The proposed pipeline is structured into two distinct phases. The first phase, Synthetic Data Generation, involves creating high-fidelity synthetic data samples that closely mimic real-world scenarios, thereby augmenting the training dataset without extensive manual intervention. This phase establishes a strong foundational dataset that improves initial model performance and generalization. The second phase, Active Learning, utilizes AL techniques to selectively identify and label the most informative real-world data points, optimizing labeling efforts and reducing human labor. By focusing on these strategically chosen samples, the pipeline maximizes data acquisition efficiency and minimizes associated costs.

## Active Learning Methods Comparison

This comprehensive figure compares all three Uncertainty Sampling strategies—Average, Max, and Sum against Random Sampling. It highlights the strengths and limitations of each method, showing how Uncertainty Sampling methods, particularly Average and Max, significantly accelerate model learning compared to Random Sampling, with the Sum method showing its value in extended training scenarios.

![AL Methods Comparison](assets/al_methods_comparison.png)


## All Classes Comparison

This figure provides a comprehensive comparison of three Uncertainty Sampling strategies—Average, Max, and Sum—against Random Sampling. The analysis is conducted across eight distinct classes, illustrating the performance differences of each strategy in handling various class-specific challenges.

![All Classes Comparison](assets/all_classes_comparison.png)

### Comparison of Random Sampling and Uncertainty Sampling

The top row presents images obtained through Random Sampling (RS), while the bottom row features images acquired using Uncertainty Sampling (US). Training sample sizes range from 50 to 200 samples.

<!-- First Row: Random Sampling Images -->
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

<!-- Space between rows -->
<br/>

<!-- Second Row: Uncertainty Sampling Images -->
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
