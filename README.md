# certified-robustness-reproduction

In this project, we reproduced the paper “On the Scalability of Certified Adversarial Robustness with Generated Data” 📄. **Using the CIFAR-100 dataset** 🖼️, We implemented the **ListDefNet model** 🧠 to evaluate how certified adversarial robustness behaves under different conditions.

Robustness means the model’s ability to correctly classify images even when they are slightly modified or attacked ⚔️. For example, if a cat image 🐱 is blurred or distorted (an adversarial attack), it may become hard for humans to recognize, and the model might wrongly predict it as a dog 🐶.

The paper mainly studies certified adversarial robustness, which means giving a guarantee that the model will still make correct predictions even under certain attacks ✅. It also focuses on scalability 📈, which refers to how well these defense methods perform when the dataset size or model complexity increases.
