# DP3 - Component-Aware Browsing of Musi# Component-Aware Browsing of Music

**Alexandra Blais**

---

## ABSTRACT

Music collections are usually browsed by metadata or whole-track similarity, which treats a song as a single mixed sound. This project builds a tool for browsing music by its individual components: a 30-second excerpt is separated into stems (vocals, drums, bass, guitar, piano, and other) with a pretrained Demucs model, and the user selects one stem to find songs containing a similar component. This report describes the current status. The separation and beat-synchronous feature pipeline is implemented and is being run on a genre-balanced subset of FMA and on up to about 1,000 chart-topping song previews. A first blind listening test comparing chroma, MFCC, and combined descriptors showed that "similarity" is not one quantity: raters weigh timbre, notes, and timing differently. We therefore define similarity per perceptual facet (timbre, notes/harmony, rhythm). Given the scope change and deadline, structure-based browsing is moved to future work, and evaluation is a single-rater listening pilot supported by objective diagnostics.

---

## 1. INTRODUCTION

Music collections are usually browsed by metadata or whole-track similarity, which treats a song as a single mixed sound. A user, however, often remembers or wants a particular element: a drum pattern, a bass line, a vocal style. Separating a song into its components lets a user ask for songs that share a type of component, such as a similar beat or a similar bass sound, not just a similar overall sound. This project tests how well automatic source separation supports that kind of search.

The application will work as follows. The user uploads a 30-second song excerpt and the application separates it into its components (stems): vocals, drums, bass, guitar, piano, and other. The user can listen to each isolated component, then select one component to browse songs containing a similar component (for example, selecting the drums in Maroon 5's "Misery" and finding songs with a similar drum set). Restricting the selection to one structural segment of the song (for example, the intro) using a self-similarity matrix and detected segment boundaries is left as future work.

The project originally centered on structure-based browsing and has since shifted toward component-level search. The shift reflects a personal use case (hearing a song, loving one component, and wanting other songs with a similar-sounding one) and a perceived gap in component-similarity software (to the author's knowledge).

The primary goal is to build and evaluate a system that separates songs into their components using a pretrained source separation model (Demucs), lets the user listen to the components, and supports similarity search at the component level. This report covers related work (Section 2), the method (Section 3), the data (Section 4), the status of each objective and preliminary results (Section 5), and lessons learned and the revised plan (Section 6).

---

## 2. RELATED WORK

Demucs separates music into stems by combining spectrogram and waveform processing [1], and its later versions add transformer layers [2]. In this project, a pretrained Demucs model is used as is, with the six-stem output, and the contribution is using its output for retrieval.

Browsing collections by audio similarity on a 2D map was introduced by Pampalk et al. [3] and extended by Pampalk and Goto [4]. Khulusi et al. [5] survey music visualization. Maps here use UMAP [6] over song-level embeddings built from pooled MFCC, chroma, and rhythm statistics, with pretrained embeddings such as MERT [7] as an advanced extension.

Music similarity is subjective. Berenzweig et al. [8] compare acoustic similarity measures with human judgments, and Flexer [9] shows that disagreement between raters limits how well any similarity measure can be evaluated. This motivates defining similarity per perceptual facet (timbre, notes/harmony, rhythm) rather than as a single score, and it is why the evaluation here is limited to a small pilot plus objective diagnostics.

For the future-work extension to structure, music structure analysis is surveyed by Paulus et al. [10] and Nieto et al. [11]. Foote [12] introduced novelty-based segmentation from the self-similarity matrix. Later methods use spectral clustering [13], time-series structure features [14], and convolutional networks for boundaries [15], and Wang et al. [16] predict structural functions such as chorus and verse. Annotated data comes from the Harmonix Set [17], and boundary evaluation would use mir\_eval [18].

The tools and data used so far are librosa [19] and FMA [20].

---

## 3. METHOD

Figure 1 shows the pipeline.

```
┌─────────────────────────────────────────┐
│            Audio (30 s)                 │
│                  ↓                      │
│       Demucs, six stems (FLAC)          │
│                  ↓                      │
│        Beat tracking on the mix         │
│                  ↓                      │
│     Chroma, MFCC, rhythm per stem       │
│                  ↓                      │
│            Pooled embeddings            │
│                  ↓                      │
│   Per-stem index, UMAP map, interface   │
└─────────────────────────────────────────┘
```

**Figure 1.** Pipeline overview: separation, beat-synchronous features, pooled embeddings, index, and interface.

### 3.1 Separation

Each track is converted to the model's format (44.1 kHz stereo), standardized, separated with a pretrained Demucs model (six stems), and de-standardized. Stems are scaled down only if they would clip. To check faithfulness we compute the relative reconstruction residual

$$
r = \frac{\left\lVert \sum_{s} \hat{x}_s - x \right\rVert}{\lVert x \rVert}, \tag{1}
$$

where $\hat{x}_s$ are the stems and $x$ is the mix ($r = 0$ for a perfect reconstruction). Stems are stored on disk as 16-bit FLAC files. Writes are atomic, so an interrupted run never leaves a half-written entry. The guitar and piano stems are known to be less reliable than the other four, so they are kept in the pipeline but excluded from the listening pilot.

### 3.2 Beat-Synchronous Features

Feature extraction does not need the model's sample rate, so all analysis uses mono audio at a reduced rate to increase processing speed. Beats are tracked once on the full mix, so that every stem shares the same beat grid. For each stem we compute chroma, which shows the pitch classes present (notes/harmony), and MFCCs without the 0th coefficient, which describe timbre. Both features are aggregated between consecutive beats (median for chroma to control for outliers, mean for MFCC). A rhythm descriptor will be added so that all three facets are covered. We expect to built it from the tempo and onset statistics pooled in the same way as the existing features. A stem whose level relative to the mix,

$$
L = 20 \log_{10} \frac{\mathrm{RMS}(\text{stem})}{\mathrm{RMS}(\text{mix})}, \tag{2}
$$

falls below a threshold is marked silent and skipped, since a near-empty stem would only add noise to the results. The fraction of stems flagged silent, per stem type, is reported as a separation indicator together with the residual $r$.

### 3.3 Pooled Embeddings

A beat-level feature matrix $F \in \mathbb{R}^{d \times n}$ is collapsed into one fixed-size vector, independent of the stem's length, by concatenating the mean and standard deviation of each feature across beats:

$$
\mathbf{e} = \left[ \mu(F) ; \sigma(F) \right] \in \mathbb{R}^{2d}. \tag{3}
$$

This gives 24 numbers for chroma ($d = 12$) and 38 for MFCC ($d = 19$). The rhythm descriptor is pooled in the same way. A full-mix embedding is computed from the mix with the same procedure, so that it serves both as the song-level embedding and as the full-mix baseline. Vectors are stored raw, and standardization and combination happen when the search index is built. Because the rhythm descriptor and the full-mix embedding are added before the batch is launched, extraction runs only once over the collection.

### 3.4 Descriptors and Search

Each block $b$ (chroma, MFCC, rhythm) is standardized per dimension over all stems of the same type and weighted by $1/\sqrt{d_b}$, where $d_b$ is the block's length, so that the blocks count equally in the combined descriptor. Stems are compared with the cosine distance:

$$
\mathbf{e}_{\text{comb}} = \left[ \tfrac{\mathbf{z}_{\text{chroma}}}{\sqrt{24}} ; \tfrac{\mathbf{z}_{\text{mfcc}}}{\sqrt{38}} ; \tfrac{\mathbf{z}_{\text{rhythm}}}{\sqrt{d_{\text{rhythm}}}} \right], \tag{4}
$$

$$
d(\mathbf{a}, \mathbf{b}) = 1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \lVert \mathbf{b} \rVert}. \tag{5}
$$

Stems are compared only with stems of the same type (for example, drums to drums) using cosine nearest neighbours. Which facet descriptors, or combinations of them, are used for each stem type is decided in the facet-determination step (Section 3.6). Song-level search uses the full-mix embedding, and the 2D map applies UMAP [6] to the song-level embeddings.

### 3.5 Pipeline Infrastructure

One script runs the download, separate, extract, table, timing, and listening stages for the provided dataset. Each stage skips tracks that are already processed, failures are logged per track with a one-line description of where the error was raised, and per-track processing times are recorded in CSV logs. The Harmonix audio downloader is already written and is kept as ready infrastructure for the future-work structure evaluation.

### 3.6 Evaluation

Because none of the collections provides isolated stems or similarity judgments, and because rater disagreement limits any similarity evaluation [9], the evaluation is deliberately small:

- **Separation quality:** a listening check on a sample of tracks across genres (including electronic music) with documented failure cases, plus the two automatic indicators above (residual and fraction of silent stems, per stem type).
- **Facet determination (reduced):** objective diagnostics per facet (tempo difference for rhythm, pitch-class agreement for notes/harmony, and timbre distance between the query and its neighbours) to check that each descriptor captures its facet. A per-stem facet priority is fixed in advance from the literature.
- **Component-level retrieval:** a blind single-rater listening pilot on sampled query stems for drums, bass, and vocals, rated 1–5, comparing the facet descriptors and their combinations with a random baseline and a full-mix baseline.
- **Cost:** processing time per song (including separation and facet extraction) and query time, recorded but not benchmarked.

---

## 4. DATA

### 4.1 Datasets

Table 1 summarizes the collections.

- **FMA:** main collection for the browsing index. A genre-balanced subset of the small subset (8,000 clips) is used, sized by the measured separation time per song so that processing finishes in a few unattended runs [20].
- **Chart-topping song previews:** FMA tracks are largely obscure, so up to about 1,000 chart-topping songs (as of November 2026) from various genres were added as 30-second previews from the iTunes Store API. This lets the application return songs that most users recognize. These previews have no genre labels and are used for research and demonstration only. If separation time is prohibitive, the number of previews is reduced before the FMA subset is.
- **Separation and similarity ground truth:** none of the collections provides isolated stems or similarity judgments, so separation and retrieval quality are checked by listening tests on a sample of tracks, supported by objective diagnostics.
- **Future work, structure ground truth:** FMA has no structure annotations, so boundary evaluation would use the Harmonix Set [17] (912 pop tracks with human-annotated beats and functional segment labels). Harmonix does not distribute audio, so audio would be obtained from the YouTube links in its metadata; tracks with dead links or poor alignment scores (below 0.95) are excluded.
- **Future work, additional collections:** MTG-Jamendo and Music4All for generalization, currently not used.

| Collection | Role | Status |
|---|---|---|
| FMA (balanced subset of fma\_small) | Main index, genre labels | In use |
| Chart previews (iTunes) | Recognizable songs, no genres | In use |
| Harmonix Set | Structure ground truth | Future (downloader ready) |
| MTG-Jamendo, Music4All | Generalization | Future |

**Table 1.** Datasets and their status.

---

## 5. STATUS OF PERFORMANCE INDICATORS

### 5.1 Status

Table 2 lists every performance indicator (PI) and its current status. Basic indicators are required, expected indicators are the target, and advanced indicators are stretch goals attempted only if time remains.

| Objective | PI | Level | Description | Status |
|---|---|---|---|---|
| 1. Separation and component analysis | PI1 | Basic | Separate songs into six stems with Demucs and cache them on disk | Done |
| | PI2 | Basic | Beat-synchronous chroma and MFCC per stem, pooled into one embedding per stem | Done |
| | PI3 | Expected | Add rhythm descriptor and full-mix embedding, run the pipeline once on the FMA subset and previews, record time per song, check separation quality | In progress |
| | PI4 | Advanced | Compare pooled facet descriptors with MERT or musicnn | Not started |
| 2. Similarity search | PI1 | Basic | Song-level embedding (also the full-mix baseline), nearest-neighbour search, UMAP map | Not started |
| | PI2 | Basic | Per-stem index and same-type nearest-stem search | Not started |
| | PI3 | Expected | Facet determination per stem type and component-level retrieval evaluation (single-rater pilot, diagnostics, random and full-mix baselines) | Partial (first listening test done) |
| | PI4 | Advanced | Multi-rater, per-facet listening study with rater agreement | Not started |
| 3. Interface | PI1 | Basic | Streamlit app: upload a 30-second excerpt, separate, show waveforms | Not started |
| | PI2 | Basic | Playback of each stem and the full mix | Not started |
| | PI3 | Expected | Component selection, nearest songs on the 2D map, audio preview of the matched component | Not started |
| | PI4 | Advanced | User-selectable facet weighting in the search | Not started |

**Table 2.** Status of performance indicators.

### 5.2 Preliminary Results

A first attempt at a blind listening test compared chroma, MFCC, and combined descriptors for stem retrieval. The rater was unable to assign a similarity rating for most inputs–the rater could not stick to one notion of similarity: depending on the stem and the query, they weighed timbre, notes, and timing differently, so no single descriptor was best across the board. This result led to the facet-based definition of similarity and the added rhythm descriptor goal. Quantitative results (separation indicators, processing time per song, facet diagnostics, and pilot ratings) will be added once the batch and indexes are complete.

---

## 6. LESSONS LEARNED AND REVISED PLAN

### 6.1 Lessons Learned

- Similarity is not one quantity. A single descriptor or score cannot serve all stem types, so similarity is now defined per facet (timbre, notes/harmony, rhythm).
- Separation and extraction are the most expensive steps, so the descriptors must be final before the batch is launched. This is why the rhythm descriptor and the full-mix embedding are added first and extraction runs only once.
- The original scope (structure analysis, Harmonix evaluation, three collections, a multi-rater study) was too large given the time contraints, due to this being an individual project. Structure and extra collections moved to future work, and the evaluation was reduced to a single-rater pilot plus objective diagnostics.
- The guitar and piano stems are less reliable in the six-stem model, so they are kept in the pipeline but excluded from the pilot.

### 6.2 Revised Plan

- **Nov 16–22:** measure seconds per song and choose the collection size; add the rhythm descriptor and full-mix embedding; launch the batch; build the stem-level and song-level indexes.
- **Nov 23–29:** facet determination per stem type (literature, diagnostics, 10-minute listening pilot); UMAP map; first version of the app (upload, waveforms, stem playback).
- **Nov 30–Due date:** component search in the app with audio preview and the map; collect numbers and make figures; write the report.

### 6.3 Future Work

- Self-similarity matrices and segment boundary detection by novelty peak picking, evaluated with mir\_eval [18] (F-measure at 0.5 s and 3 s tolerance) on the Harmonix Set.
- Segment-restricted component search: embed a stem within a detected segment, add structural attributes (segment duration, relative position), evaluate precision@k against annotated section labels, and display a colour-coded segment timeline.
- MTG-Jamendo and Music4All as additional collections.
- User-contributed 30-second snippets that grow the collection.

---

## 7. CONCLUSION

The separation and beat-synchronous feature pipeline is working. The first evaluation showed that component similarity cannot be defined by a single score, so the project now treats similarity as facet-dependent (timbre, notes/harmony, rhythm) and will test which descriptor captures which facet for each stem type. The main remaining work is the rhythm descriptor and full-mix embedding, the batch run, the search indexes, the 2D map, the reduced evaluation, and the interface. Structure-based browsing is left as future work.

---

## REFERENCES

[1] A. Défossez, "Hybrid spectrogram and waveform source separation," in *Proc. ISMIR 2021 Workshop on Music Source Separation*, 2021.

[2] S. Rouard, F. Massa, and A. Défossez, "Hybrid transformers for music source separation," in *Proc. IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)*, 2023.

[3] E. Pampalk, A. Rauber, and D. Merkl, "Content-based organization and visualization of music archives," in *Proc. 10th ACM Int. Conf. on Multimedia*, pp. 570–579, 2002.

[4] E. Pampalk and M. Goto, "MusicRainbow: A new user interface to discover artists using audio-based similarity and web-based labeling," in *Proc. 7th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2006.

[5] R. Khulusi, J. Kusnick, C. Meinecke, C. Gillmann, J. Focht, and S. Jänicke, "A survey on visualizations for musical data," *Computer Graphics Forum*, vol. 39, no. 6, pp. 82–110, 2020.

[6] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," *arXiv:1802.03426*, 2018.

[7] Y. Li *et al.*, "MERT: Acoustic music understanding model with large-scale self-supervised training," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2024.

[8] A. Berenzweig, B. Logan, D. P. W. Ellis, and B. Whitman, "A large-scale evaluation of acoustic and subjective music-similarity measures," *Computer Music Journal*, vol. 28, no. 2, pp. 63–76, 2004.

[9] A. Flexer, "On inter-rater agreement in audio music similarity," in *Proc. 15th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2014.

[10] J. Paulus, M. Müller, and A. Klapuri, "State of the art report: Audio-based music structure analysis," in *Proc. 11th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2010.

[11] O. Nieto, G. J. Mysore, C. Wang, J. B. L. Smith, J. Schlüter, T. Grill, and B. McFee, "Audio-based music structure analysis: Current trends, open challenges, and applications," *Trans. Int. Society for Music Information Retrieval*, vol. 3, no. 1, pp. 246–263, 2020.

[12] J. Foote, "Automatic audio segmentation using a measure of audio novelty," in *Proc. IEEE Int. Conf. on Multimedia and Expo (ICME)*, pp. 452–455, 2000.

[13] B. McFee and D. P. W. Ellis, "Analyzing song structure with spectral clustering," in *Proc. 15th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2014.

[14] J. Serrà, M. Müller, P. Grosche, and J. L. Arcos, "Unsupervised music structure annotation by time series structure features and segment similarity," *IEEE Trans. Multimedia*, vol. 16, no. 5, pp. 1229–1240, 2014.

[15] K. Ullrich, J. Schlüter, and T. Grill, "Boundary detection in music structure analysis using convolutional neural networks," in *Proc. 15th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2014.

[16] J.-C. Wang, Y.-N. Hung, and J. B. L. Smith, "To catch a chorus, verse, intro, or anything else: Analyzing a song with structural functions," in *Proc. IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)*, 2022.

[17] O. Nieto, M. C. McCallum, M. E. P. Davies, A. Robertson, A. Stark, and E. Egozcue, "The Harmonix Set: Beats, downbeats, and functional segment annotations of Western popular music," in *Proc. 20th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2019.

[18] C. Raffel *et al.*, "mir\_eval: A transparent implementation of common MIR metrics," in *Proc. 15th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2014.

[19] B. McFee *et al.*, "librosa: Audio and music signal analysis in Python," in *Proc. 14th Python in Science Conf.*, 2015.

[20] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson, "FMA: A dataset for music analysis," in *Proc. 18th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2017.0. Flexer, A. (2014). On inter-rater agreement in audio music similarity. *Proceedings of the 15th ISMIR*.
