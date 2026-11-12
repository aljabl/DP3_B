# DP3 - Component-Aware Browsing of Music

## Project Overview

This project builds a tool for browsing music by its individual components. The user uploads a song and the application separates it into its components (stems): vocals, drums, bass and other instruments. The user can listen to each isolated component, then (a) select one component to browse songs containing a similar component (for example, selecting the drums in Maroon 5's "Misery", and finding songs with a similar drum set), or (b) optionally restrict the selection to one structural segment of the song (for example, the intro) using a self-similarity matrix and detected segment boundaries. Music is used in place of sound effects.

## Problem Statement

**Primary Goal:**
Build and evaluate a system that separates songs into their components using a pretrained source separation model (Demucs), lets the user listen to and select a component, and supports similarity search at the component level.

**Additional Goals:**

- Separate songs into stems and compute a descriptor for each stem.
- Compare feature types (chroma, MFCC and their combination) for describing components.
- Define song-level and component-level descriptors for retrieval.
- Place songs on a 2D map so that similar songs are located nearby.
- Detect segment boundaries from a self-similarity matrix so that a component can be selected within a specific section of a song.
- Record computation time per song and per query.

## Motivation

Music collections are usually browsed by metadata or whole-track similarity, which treats a song as a single mixed sound. A user, however, often remembers or wants a particular element: a drum pattern, a bass line, a vocal style. Separating a song into its components lets a user ask for songs that share a type of component, such as a similar beat or a similar bass sound, not just a similar overall sound. This project tests how well automatic source separation supports that kind of search.

## Dataset Description

- **FMA:** main collection for the browsing index. The small subset (8,000 clips) is used; if separation time is prohibitive, a genre-balanced subset of it is used. <https://github.com/mdeff/fma>
- **MTG-Jamendo and Music4All:** optional additional collections for generalization, used only if time allows. <https://mtg.github.io/mtg-jamendo-dataset/>, <https://sites.google.com/view/contact4music4all>
- **Structure ground truth (advanced goals only):** FMA has no structure annotations, so boundary evaluation uses the Harmonix Set (912 pop tracks with human-annotated beats and functional segment labels). Harmonix does not distribute audio, so audio is obtained from the YouTube links in its metadata; tracks with dead links or poor alignment scores (below 0.95) are excluded. <https://github.com/urinieto/harmonixset>
- **Separation ground truth:** neither collection provides isolated stems, so separation quality is checked by listening on a sample of tracks, and retrieval quality is evaluated with genre labels.

## Proposed Methodology

- Audio preprocessing
- Source separation into stems with Demucs
- Beat-synchronous feature extraction for each stem
- Component-level and song-level descriptors
- Nearest-neighbour search and 2D map
- Interactive web interface
- Self-similarity matrix and boundary detection (advanced)

## Tools and Technologies

- **Source separation:** Demucs (<https://github.com/adefossez/demucs>), a pretrained model used as is (no training), with the default four-stem output (vocals, drums, bass, other)
- **Python:** librosa, mir_eval, scikit-learn, umap-learn
- **Embeddings:** pooled chroma/MFCC statistics per stem and per song; a pretrained model (MERT or musicnn) is an optional extension
- Kadenze course notebooks (clustering and visualization) as baseline code
- **Interface:** Plotly or Bokeh with Flask or Streamlit; audio playback in the browser

## Evaluation Metrics

- Separation quality: listening check on a sample of tracks across genres, with documented failure cases
- Precision@k of component-level retrieval against genre labels, for each stem type, compared with a full-mix baseline
- Boundary detection F-measure (0.5 s and 3 s tolerance) and precision@k of segment-level retrieval against annotated section labels (advanced)
- Processing time per song (including separation time) and query time (recorded, not benchmarked)

## Timeline

- **Weeks 1-2:** Dataset preparation (including Demucs setup, separation of the FMA subset, and, for the advanced goals, Harmonix audio download and alignment check), beat-synchronous feature extraction for each stem.
- **Weeks 3-4:** Stem and song descriptors, component index and retrieval, evaluation against genre labels, 2D map.
- **Week 5:** Interactive interface (upload, stem playback, component selection and search).
- **Week 6:** Advanced goals if time allows (self-similarity matrix, boundary detection, segment-restricted search), final testing, documentation, and presentation preparation.

## Team

Individual project; one person covers all components.

## Related Work

Demucs separates music into stems by combining spectrogram and waveform processing (Défossez, 2021), and its later versions add transformer layers (Rouard et al., 2023). Here a pretrained Demucs model is used as a fixed component, and the project's contribution is using its output for retrieval.

Browsing collections by audio similarity on a 2D map was introduced by Pampalk et al. (2002) and extended by Pampalk & Goto (2006); Khulusi et al. (2020) survey music visualization. Maps here use UMAP (McInnes et al., 2018) over song-level embeddings built from pooled chroma and MFCC statistics, with pretrained embeddings such as MERT (Li et al., 2024) as an optional extension.

For the advanced goals, music structure analysis is surveyed by Paulus et al. (2010) and Nieto et al. (2020). Foote (2000) introduced novelty-based segmentation from the self-similarity matrix. Later methods use spectral clustering (McFee & Ellis, 2014), time-series structure features (Serra et al., 2014) and convolutional networks for boundaries (Ullrich et al., 2014). MSAF (Nieto & Bello, 2016) packages several algorithms, and Wang et al. (2022) predict structural functions such as chorus and verse. Annotated data comes from SALAMI (Smith et al., 2011) and the Harmonix Set (Nieto et al., 2019).

Tools and data: librosa (McFee et al., 2015), mir_eval (Raffel et al., 2014), FMA (Defferrard et al., 2017), and Müller (2015) for background on feature extraction and self-similarity matrices.

## Alexandra Blais

### Objective 1: Implement an automatic source separation and component analysis pipeline

- [ ] **PI1 (basic):** Load and preprocess audio, separate each song into stems (vocals, drums, bass, other) with a pretrained Demucs model, and cache the stems on disk
- [ ] **PI2 (basic):** Extract beat-synchronous chroma and MFCC features for each stem and pool them into one embedding per stem
- [ ] **PI3 (expected):** Run the pipeline on the FMA subset, check separation quality by listening on a sample of tracks across genres (including electronic music) and document failure cases, and compare chroma, MFCC and their combination as stem descriptors while recording processing time per song
- [ ] **PI4 (advanced):** Compute self-similarity matrices and detect segment boundaries by novelty peak picking, and evaluate boundary F-measure (0.5 s and 3 s tolerance) with mir_eval on the Harmonix Set (tracks with usable YouTube audio), so that a component can be selected within a section of a song

### Objective 2: Develop song-level and component-level similarity search

- [ ] **PI1 (basic):** Compute a song-level embedding and nearest-neighbour search over the FMA subset, plus a UMAP map of the collection
- [ ] **PI2 (basic):** Index the embedding of every stem, and search for the nearest stems of the same type (for example, drums to drums)
- [ ] **PI3 (expected):** Evaluate component-level retrieval with precision@k against genre labels for each stem type, and compare it with a full-mix baseline
- [ ] **PI4 (advanced):** Restrict component search to a time segment by embedding a stem within a detected segment, add structural attributes (segment duration, relative position), and evaluate precision@k against annotated section labels on Harmonix

### Objective 3: Build an interactive component-browsing interface

- [ ] **PI1 (basic):** Create a web app that accepts an uploaded song, separates it into stems, and displays the waveform of the full mix and of each stem
- [ ] **PI2 (basic):** Add playback of each isolated stem and of the full mix, so the user can listen to every component
- [ ] **PI3 (expected):** Implement component selection that returns the nearest songs with a similar component, shown on the 2D map with audio preview of the matched component
- [ ] **PI4 (advanced):** Display the self-similarity matrix and a colour-coded segment timeline, let the user select a component within a segment, and return songs with a similar component, with the matched segment highlighted and playable

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
20. Défossez, A. (2021). Hybrid spectrogram and waveform source separation. *Proceedings of the ISMIR 2021 Workshop on Music Source Separation*.
21. Rouard, S., Massa, F., & Défossez, A. (2023). Hybrid transformers for music source separation. *Proceedings of ICASSP 2023*.
