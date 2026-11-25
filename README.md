# YouTube Clickbait Detector: Enhanced

A research project that replicates a published YouTube clickbait classifier (Vierti et al.) and tests whether adding **what viewers say** (comment sentiment) and **what the video actually contains** (how well the title matches the transcript) improves detection.

Built as a team research project for the Research Methodology in Computer Science course at BINUS University. The full write-up is in [`report/report.pdf`](report/report.pdf), in IEEE format.

## Results

All three models were evaluated once on the same held-out test set of 1,145 videos (573 non-clickbait, 572 clickbait).

| Model | Features | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| 1. Baseline SVM (replication) | Title Word2Vec + engagement stats | 95.8% | 96.5% | 95.1% | 0.958 |
| 2. Enhanced SVM | Baseline + comment sentiment + title-transcript similarity | **96.2%** | 96.3% | **96.0%** | **0.961** |
| 3. Enhanced Random Forest | Same as Model 2 | 93.3% | 94.1% | 92.3% | 0.932 |

The enhanced SVM beat the replicated baseline, mainly by catching more clickbait (recall went from 95.1% to 96.0%).

## How it works

### Stage 1: Data collection and filtering

Starting from the Vierti et al. dataset of about 34,000 labeled YouTube videos:

1. **Transcript availability:** check every video with `youtube-transcript-api`. If that fails, fall back to the YouTube Data API.
2. **Comment availability:** check comment counts with batched YouTube Data API `videos.list` calls (50 videos per call), which is far cheaper on quota than per-video requests.
3. **Filter and balance:** keep videos that have both a transcript and enough comments, then trim to a 50/50 clickbait split. This gives a working set of 5,722 videos.
4. **Extract:** download transcripts and up to 100 top comments per video.

Collection took about two days. Requests were paced with cooldowns to avoid IP blocks, and the scripts resume from where they stopped.

### Stage 2: Feature blocks

| Block | Features | Tools |
| --- | --- | --- |
| A | Title embedding + engagement (views, likes, dislikes, comments) | Word2Vec (gensim) |
| B | Sentiment of viewer comments | VADER |
| C | Semantic similarity between title and transcript | Sentence-BERT |

### Stage 3: Training and evaluation

- 80/20 train/test split, with the split indices saved so every model sees the same test set
- **Model 1** uses the original paper's hyperparameters (C = 3.7, γ = 4.1) to keep the replication faithful.
- **Models 2 and 3** are tuned with 5-fold cross-validated grid search on the training set, optimizing F1. The original hyperparameters were included in the search grid so they stayed reachable.

## Try it

```bash
cd demo
pip install numpy pandas gensim scikit-learn
python build_artifacts.py     # run once
python predict.py <youtube_video_id>
python predict.py --random --clickbait
```

The demo prints the video's title and channel, the predicted label and confidence, and the engagement and sentiment features behind the prediction.

## Project structure

```
Stage_1_Extraction_Validation_Filtering/   Data collection, availability checks, filtering
Stage_2_SVM_Vader_SBert/                   Feature extraction (blocks A, B, C)
Stage_3_Training/                          Training scripts, evaluation, results per model
demo/                                      Command-line prediction demo
report/                                    IEEE-format paper (LaTeX source and PDF)
```

## Tech stack

Python, pandas, scikit-learn (SVM, Random Forest, GridSearchCV), gensim (Word2Vec), Sentence-BERT, VADER, YouTube Data API v3, youtube-transcript-api, LaTeX

## Team

- **Yonathan Eivan Riyadi:** team lead; built the data collection pipeline
- <!-- Teammate name: role -->

## Reference

Vierti et al., the original clickbait detection study replicated here. Full citation in [`report/bibliography.bib`](report/bibliography.bib).
