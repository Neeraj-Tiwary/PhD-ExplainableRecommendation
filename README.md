# Semantic Explainable Recommender with Knowledge Graph Embedding and Deep Reinforcement Learning
This repository contains the source code of the Explainable Recommendation

## Datasets
Two Amazon datasets (Amazon_Beauty, Amazon_Cellphones) are available in the "data/" directory and the split is consistent with [1].
All four datasets used in this paper can be downloaded [here](https://drive.google.com/uc?export=download&confirm=Tiux&id=1CL4Pjumj9d7fUDQb1_leIMOot73kVxKB).

## Requirements
- Python >= 3.6
- PyTorch = 1.0


## How to run the code
1. Proprocess the data first:
```bash
python preprocess.py --dataset <dataset_name>
```
"<dataset_name>" should be one of "cd", "beauty", "cloth", "cell" (refer to utils.py).

2. Train knowledge graph embeddings (TransE in this case):
```bash
python train_transe_model.py --dataset <dataset_name>
```

3. Train RL agent:
```bash
python train_RL_agent.py --dataset <dataset_name>
```

4. Evaluation
```bash
python test_RL_agent.py --dataset <dataset_name> --run_path True --run_eval True
```
If "run_path" is True, the program will generate paths for recommendation according to the trained policy.
If "run_eval" is True, the program will evaluate the recommendation performance based on the resulting paths.

## Streamlit Application

A simple Streamlit web application (`app.py`) is provided for interactive exploration and demonstration of the recommendation system.

### Running the Streamlit App

1. Install Streamlit if not already installed:
    ```bash
    pip install streamlit
    ```

2. Start the application:
    ```bash
    streamlit run app.py
    ```

3. Open the provided local URL in your browser to interact with the app.

You can customize `app.py` to visualize recommendations, upload datasets, or display explainability paths as needed.


## References
### Journal Publications
1. N. Tiwary, S. A. Mohd Noah, F. Fauzi and T. S. Yee, Max Explainability Score–A quantitative metric for explainability evaluation in knowledge graph-based recommendations, Computers and Electrical Engineering, vol. 116, p. 109190, 2024, https://doi.org/10.1016/j.compeleceng.2024.109190
2. N. Tiwary, S. A. M. Noah, F. Fauzi and T. S. Yee, A Review of Explainable Recommender Systems Utilizing Knowledge Graphs and Reinforcement Learning, in IEEE Access, vol. 12, pp. 91999-92019, 2024, Doi: 10.1109/ACCESS.2024.3422416
3. S. M. Al-Ghuribi, S. A. Mohd Noah, T. N. Mohammed, N. Tiwary and N. I. Y. Saat, A Comparative Study of Sentiment-Aware Collaborative Filtering Algorithms for Arabic Recommendation Systems, IEEE Access, vol. 12, pp. 174441-174454, 2024, doi: 10.1109/ACCESS.2024.3489658
4. Tiwary, N., Mohd Noah, S., Fauzi, F., Yee, T., & Al-Ghuribi, S, Enhancing Recommender Systems with Deep reinforcement Learning and Knowledge Graph Embeddings, Malaysian Journal of Computer Science, 2025, In Press.

### Conference Contributions
1. N. Tiwary, S. A. M. Noah, F. Fauzi and T. S. Yee, Advancing Recommender Systems with Deep Reinforcement Learning, in 16th IEEE International Conference on Knowledge and Systems Engineering (KSE 2024), Kuala Lumpur, Malaysia, 5-7 Nov 2024
2. Neeraj Tiwary, Shahrul Azman Mohd Noah, Fariza Fauzi, Steffen Stabb, Explainable recommender – Implementation approaches, In E- Proceedings of the 5th International Multi-Conference on Artificial Intelligence Technology, (MCAIT 2021) Artificial Intelligence in the 4th Industrial Revolution, Kuala Lumpur, Malaysia, 4-5 Aug 2021, pp 158-161, (https://www.ftsm.ukm.my/mcait2021/file/The-5th-MCAIT2021-eProceeding.pdf)