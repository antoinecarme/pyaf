## TimeSEries Foundational Models

To be investigated.

Open source models for general purpose.

PyAF can always pick some new signal transformations from these models.

### IBM's TinyTimeMixer

TinyTimeMixers (TTMs) are compact pre-trained models for Multivariate Time-Series Forecasting, open-sourced by IBM Research. With less than 1 Million parameters, TTM introduces the notion of the first-ever “tiny” pre-trained models for Time-Series Forecasting.

TTM outperforms several popular benchmarks demanding billions of parameters in zero-shot and few-shot forecasting. TTMs are lightweight forecasters, pre-trained on publicly available time series data with various augmentations. TTM provides state-of-the-art zero-shot forecasts and can easily be fine-tuned for multi-variate forecasts with just 5% of the training data to be competitive. Refer to our paper for more details.

The current open-source version supports point forecasting use-cases ranging from minutely to hourly resolutions (Ex. 10 min, 15 min, 1 hour, etc.)

Note that zeroshot, fine-tuning and inference tasks using TTM can easily be executed in 1 GPU machine or in laptops too!!

https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1

https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer

### Google TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google Research for time-series forecasting.

A decoder-only foundation model for time-series forecasting
Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou

    Motivated by recent advances in large language models for Natural Language Processing (NLP), we design a time-series foundation model for forecasting whose out-of-the-box zero-shot performance on a variety of public datasets comes close to the accuracy of state-of-the-art supervised forecasting models for each individual dataset. Our model is based on pretraining a patched-decoder style attention model on a large time-series corpus, and can work well across different forecasting history lengths, prediction lengths and temporal granularities.
    

https://github.com/google-research/timesfm

https://huggingface.co/google/timesfm-1.0-200m

https://arxiv.org/abs/2310.10688

### Carnegie Mellon AntonLab's Moment


MOMENT is a family of foundation models for general-purpose time-series analysis. The models in this family (1) serve as a building block for diverse time-series analysis tasks (e.g., forecasting, classification, anomaly detection, and imputation, etc.), (2) are effective out-of-the-box, i.e., with no (or few) task-specific exemplars (enabling e.g., zero-shot forecasting, few-shot classification, etc.), and (3) are tunable using in-distribution and task-specific data to improve performance.

    We introduce MOMENT, a family of open-source foundation models for general-purpose time-series analysis. Pre-training large models on time-series data is challenging due to (1) the absence of a large and cohesive public time-series repository, and (2) diverse time-series characteristics which make multi-dataset training onerous. Additionally, (3) experimental benchmarks to evaluate these models, especially in scenarios with limited resources, time, and supervision, are still in their nascent stages. To address these challenges, we compile a large and diverse collection of public time-series, called the Time-series Pile, and systematically tackle time-series-specific challenges to unlock large-scale multi-dataset pre-training. Finally, we build on recent work to design a benchmark to evaluate time-series foundation models on diverse tasks and datasets in limited supervision settings. Experiments on this benchmark demonstrate the effectiveness of our pre-trained models with minimal data and task-specific fine-tuning. Finally, we present several interesting empirical observations about large pre-trained time-series models. Our code is available anonymously at anonymous.4open.science/r/BETT-773F/. 

    Hardware Type: NVIDIA RTX A6000 GPU
    GPU Hours: 404
    Compute Region: Pittsburgh, USA
    Carbon Emission (tCO2eq):


All models were trained and evaluated on a computing cluster consisting of 128 AMD EPYC 7502 CPUs, 503 GB of RAM, and 8 NVIDIA RTX A6000 GPUs each with 49 GiB RAM. All MOMENT variants were trained on a single A6000 GPU (with any data or model parallelism).

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). MOMENT: A Family of Open Time-series Foundation Models. In International Conference on Machine Learning. PMLR.

https://arxiv.org/abs/2402.03885
https://arxiv.org/pdf/2402.03885


https://huggingface.co/AutonLab/MOMENT-1-large