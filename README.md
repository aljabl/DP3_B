# DP3 - Structure-Aware Browsing of Music

## Project Overview

This project builds a tool for browsing music by song structure. The user uploads a song and sees a visualization of its structure: a self-similarity matrix and a timeline of segments (for example intro, verse, chorus). The user can then (a) select the whole song to browse similar songs located nearby, or (b) select one structural segment to browse songs containing a similar segment. Music is used in place of sound effects.

## Problem Statement

**Primary Goal:**
Build and evaluate a system that segments songs automatically, visualizes their structure, and supports similarity search at both song level and segment level.

**Additional Goals:**

- Detect segment boundaries and label repeated sections from a self-similarity matrix.
- Compare feature types (chroma, MFCC, pretrained embeddings) for structure analysis.
- Define song-level and segment-level descriptors for retrieval.
- Place songs on a 2D map so that similar songs are located nearby.
- Measure computation time and query latency as the collection grows.

## Motivation

Music collections are usually browsed by metadata or whole-track similarity, which ignores how a song is built. Structure lets a user ask for songs that share a type of section, such as a similar chorus or breakdown, not just a similar overall sound. This project tests how well automatic structure analysis supports that kind of search.

## Dataset Description

- **FMA:** main collection for the browsing index. Small (8,000 clips) and medium (25,000 clips) subsets. <https://github.com/mdeff/fma>
- **MTG-Jamendo:** second collection for generalization. <https://mtg.github.io/mtg-jamendo-dataset/>
- **Music4All:** third collection. <https://sites.google.com/view/contact4music4all>
- **Structure ground truth:** FMA has no structure annotations, so boundary and label evaluation uses the Harmonix Set (912 pop tracks with human-annotated beats and functional segment labels). Harmonix does not distribute audio, so audio is obtained from the YouTube links in its metadata; tracks with dead links or poor alignment scores are excluded. <https://github.com/urinieto/harmonixset>

## Proposed Methodology

- Audio preprocessing
- Beat-synchronous feature extraction
- Self-similarity matrix and boundary detection
- Segment labelling by clustering
- Song-level and segment-level descriptors
- Nearest-neighbour search and 2D map
- Interactive web interface

## Tools and Technologies

- **Python:** librosa, MSAF, mir_eval, scikit-learn, umap-learn, FAISS
- **Pretrained embeddings:** MERT or musicnn
- Kadenze course notebooks (clustering and visualization) as baseline code
- **Interface:** Plotly or Bokeh with Flask or Streamlit; audio playback in the browser

## Evaluation Metrics

- Boundary detection F-measure (0.5 s and 3 s tolerance) and segment labelling F-measure
- Precision@k of song-level retrieval against genre labels
- Precision@k of segment-level retrieval against annotated section labels
- Trustworthiness of the 2D map
- Processing time per song, index build time, and query latency

## Timeline

- **Weeks 1-2:** Dataset preparation (including Harmonix audio download and alignment check), feature extraction, self-similarity matrices, first boundary detection.
- **Weeks 3-4:** Segment labelling, evaluation on annotated data, song-level and segment-level descriptors, retrieval.
- **Week 5:** 2D map and interactive interface.
- **Week 6:** Final testing, documentation, and presentation preparation.

## Team

Individual project; one person covers all components.

## Related Work

Music structure analysis is surveyed by Paulus et al. (2010) and Nieto et al. (2020). Foote (2000) introduced novelty-based segmentation from the self-similarity matrix. Later methods use spectral clustering (McFee & Ellis, 2014), time-series structure features (Serra et al., 2014) and convolutional networks for boundaries (Ullrich et al., 2014). MSAF (Nieto & Bello, 2016) packages several algorithms, and Wang et al. (2022) predict structural functions such as chorus and verse. Annotated data comes from SALAMI (Smith et al., 2011) and the Harmonix Set (Nieto et al., 2019).

Browsing collections by audio similarity on a 2D map was introduced by Pampalk et al. (2002) and extended by Pampalk & Goto (2006); Khulusi et al. (2020) survey music visualization. Maps here use UMAP (McInnes et al., 2018) over audio embeddings such as MERT (Li et al., 2024), with nearest-neighbour search by FAISS (Johnson et al., 2019).

Tools and data: librosa (McFee et al., 2015), mir_eval (Raffel et al., 2014), FMA (Defferrard et al., 2017), and Müller (2015) for background on self-similarity matrices.

## Alexandra Blais

### Objective 1: Implement an automatic music structure analysis pipeline

- **PI1 (basic):** Load and preprocess audio and extract beat-synchronous chroma and MFCC features
- **PI2 (basic):** Compute self-similarity matrices and detect segment boundaries by novelty peak picking
- **PI3 (expected):** Label repeated segments by clustering, and evaluate boundary and label F-measures with mir_eval on the Harmonix Set (tracks with usable YouTube audio)
- **PI4 (expected):** Compare chroma, MFCC and pretrained embedding features, and tune kernel size and peak-picking parameters on validation data
- **PI5 (advanced):** Compare at least two segmentation algorithms (for example novelty-based and spectral clustering via MSAF) and report accuracy and processing time per song

### Objective 2: Develop song-level and segment-level similarity search

- **PI1 (basic):** Compute a song-level embedding and nearest-neighbour search over the FMA small subset, plus a UMAP map of the collection
- **PI2 (basic):** Extract segments from detected boundaries, compute a segment embedding for each, and index all segments
- **PI3 (expected):** Add structural descriptors (segment length and label sequence, self-similarity-based descriptor) and compare retrieval against a timbre-only baseline using precision@k on genre labels
- **PI4 (expected):** Combine segment audio embeddings with structural attributes (duration, relative position, repetition count) and evaluate precision@k against annotated section labels
- **PI5 (advanced):** Scale to the FMA medium subset, compare exact and approximate (FAISS) search on recall and query time, and report index build time

### Objective 3: Build an interactive structure-browsing interface

- **PI1 (basic):** Create a web app that accepts an uploaded song and displays its waveform and self-similarity matrix
- **PI2 (basic):** Overlay a colour-coded segment timeline and add playback of a selected segment
- **PI3 (expected):** Implement whole-song selection that returns the nearest songs, shown on the 2D map with audio preview
- **PI4 (expected):** Implement segment selection that returns songs with a similar segment, with the matched segment highlighted and playable
- **PI5 (advanced):** Place uploaded songs on the existing map without recomputing it, measure agreement with a full re-layout (neighbourhood preservation), and benchmark end-to-end processing time for an upload

## References

1. Pampalk, E., Rauber, A., & Merkl, D. (2002). Content-based organization and visualization of music archives. *Proceedings of the 10th ACM International Conference on Multimedia*, 570-579.
2. Khulusi, R., Kusnick, J., Meinecke, C., Gillmann, C., Focht, J., & Jänicke, S. (2020). A survey on visualizations for musical data. *Computer Graphics Forum, 39*(6), 82-110.
3. Pampalk, E., & Goto, M. (2006). MusicRainbow: A new user interface to discover artists using audio-based similarity and web-based labeling. *Proceedings of the 7th ISMIR*.
4. Foote, J. (2000). Automatic audio segmentation using a measure of audio novelty. *Proceedings of the IEEE International Conference on Multimedia and Expo (ICME)*, 452-455.
5. Paulus, J., Müller, M., & Klapuri, A. (2010). State of the art report: Audio-based music structure analysis. *Proceedings of the 11th ISMIR*.
6. Nieto, O., Mysore, G. J., Wang, C., Smith, J. B. L., Schlüter, J., Grill, T., & McFee, B. (2020). Audio-based music structure analysis: Current trends, open challenges, and applications. *Transactions of the ISMIR, 3*(1), 246-263.
7. McFee, B., & Ellis, D. P. W. (2014). Analyzing song structure with spectral clustering. *Proceedings of the 15th ISMIR*.
8. Nieto, O., & Bello, J. P. (2016). Systematic exploration of computational music structure research. *Proceedings of the 17th ISMIR*.
9. Smith, J. B. L., Burgoyne, J. A., Fujinaga, I., De Roure, D., & Downie, J. S. (2011). Design and creation of a large-scale database of structural annotations. *Proceedings of the 12th ISMIR*.
10. Nieto, O., McCallum, M. C., Davies, M. E. P., Robertson, A., Stark, A., & Egozcue, E. (2019). The Harmonix Set: Beats, downbeats, and functional segment annotations of Western popular music. *Proceedings of the 20th ISMIR*.
11. Serra, J., Müller, M., Grosche, P., & Arcos, J. L. (2014). Unsupervised music structure annotation by time series structure features and segment similarity. *IEEE Transactions on Multimedia, 16*(5), 1229-1240.
12. Ullrich, K., Schlüter, J., & Grill, T. (2014). Boundary detection in music structure analysis using convolutional neural networks. *Proceedings of the 15th ISMIR*.
13. Wang, J.-C., Hung, Y.-N., & Smith, J. B. L. (2022). To catch a chorus, verse, intro, or anything else: Analyzing a song with structural functions. *Proceedings of ICASSP 2022*.
14. Müller, M. (2015). *Fundamentals of Music Processing*. Springer.
15. Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A dataset for music analysis. *Proceedings of the 18th ISMIR*.
16. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv:1802.03426*.
17. McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*.
18. Raffel, C., et al. (2014). mir_eval: A transparent implementation of common MIR metrics. *Proceedings of the 15th ISMIR*.
19. Li, Y., et al. (2024). MERT: Acoustic music understanding model with large-scale self-supervised training. *ICLR 2024*.
20. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data, 7*(3), 535-547.0. Li, Y., et al. (2024). MERT: Acoustic music understanding model with large-scale self-supervised training. ICLR 2024.
