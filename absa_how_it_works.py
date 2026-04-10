"""
absa_how_it_works.py
--------------------
Renders the How It Works tab content for the ABSA Streamlit application.
Import and call render() inside the tab context in absa_streamlit_app.py.

Usage:
    import absa_how_it_works
    with tab6:
        absa_how_it_works.render(sec)
"""

import streamlit as st


def render(sec):
    """
    Render all How It Works content.
    Accepts the sec() helper function from the main app so styling stays
    consistent without duplicating the helper here.
    """

    # ---- Pipeline Overview ----
    sec("Pipeline Overview")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;line-height:1.9;'
        'font-size:0.88rem;max-width:760px;">'
        'The system is a two-stage BERT pipeline. The first stage finds aspect terms in a '
        'sentence. The second stage classifies the sentiment toward each aspect. '
        'Both stages are fine-tuned from bert-base-uncased on SemEval ABSA datasets.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Stage 1 ----
    sec("Stage 1: Aspect Term Extraction (ATE)")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;line-height:1.9;font-size:0.88rem;max-width:760px;">'
        'The ATE model is a token classifier. It assigns one of three BIO labels to every word in the input sentence.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#888;'
        'margin:0.4rem 0 0.8rem 0;">'
        'Labels: &nbsp; <span style="color:#2ecc71;">B-ASP</span> (beginning of aspect) &nbsp; '
        '<span style="color:#f39c12;">I-ASP</span> (inside aspect) &nbsp; '
        '<span style="color:#555;">O</span> (not an aspect)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \text{Given sentence } S = \{w_1, w_2, \ldots, w_n\}
    \quad \text{predict} \quad
    y_i \in \{\text{B-ASP},\ \text{I-ASP},\ \text{O}\}
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'The model is trained with token-level cross-entropy loss, summed over all labelled positions '
        '(subword continuations and special tokens are masked with -100 and excluded):'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \mathcal{L}_{\text{ATE}} = -\sum_{i=1}^{n} \log P(y_i \mid S)
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#555;font-size:0.82rem;'
        'margin:0.2rem 0 0.5rem 0;">'
        'Because BERT uses WordPiece tokenization, one word can split into multiple subword tokens. '
        'The real BIO label is assigned only to the first subword of each word. All other subwords '
        'receive a mask value of -100 so they are ignored by PyTorch during loss computation.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Stage 2 ----
    sec("Stage 2: Aspect Sentiment Classification")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;line-height:1.9;font-size:0.88rem;max-width:760px;">'
        'For each extracted aspect term, the sentiment model receives a formatted input pair and '
        'predicts one of four classes.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \text{Input: } [\text{CLS}]\ a\ [\text{SEP}]\ S\ [\text{SEP}]
    \quad \Rightarrow \quad
    \hat{y} \in \{\text{positive},\ \text{negative},\ \text{neutral},\ \text{conflict}\}
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'Trained with standard cross-entropy loss over the four sentiment classes:'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \mathcal{L}_{\text{sent}} = -\sum_{k=1}^{4} y_k \log p_k
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#555;font-size:0.82rem;margin:0.2rem 0 0.5rem 0;">'
        'where y is the one-hot true label and p is the softmax output of the model. '
        'The conflict class is used when a sentence expresses contradictory opinions about '
        'the same aspect simultaneously.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Softmax ----
    sec("Softmax and Class Probabilities")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'The raw logit outputs of the model are converted to a probability distribution using softmax:'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    p_k = \frac{e^{z_k}}{\sum_{j=1}^{4} e^{z_j}}
    """)

    # ---- Weighted score ----
    sec("Weighted Sentiment Score")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'Instead of using only the hard predicted label, we compute a continuous score as the dot product '
        'of the softmax probabilities with signed class values. This captures model uncertainty better '
        'than a single label.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    s = p_{\text{pos}} \cdot (+1) \;+\; p_{\text{neg}} \cdot (-1) \;+\; p_{\text{neu}} \cdot 0 \;+\; p_{\text{con}} \cdot (-0.5)
    \quad \in [-1,\ +1]
    """)

    # ---- Vendor score ----
    sec("Vendor-Level Score")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'Aspect scores are aggregated across all reviews for a vendor using frequency weighting. '
        'Aspects mentioned more often contribute more to the final score.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    S_{\text{vendor}} = \sum_{a \in A} \frac{c_a}{C} \cdot \bar{s}_a
    \quad \text{where } c_a = \text{mentions of aspect } a, \quad C = \sum_a c_a
    """)

    # ---- Bayesian ----
    sec("Bayesian Rating Adjustment")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'A vendor with very few reviews could have a misleadingly high or low score. '
        'The Bayesian average pulls the score toward the global mean when review count is low.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    S_{\text{bayes}} = \frac{v \cdot S_{\text{vendor}} + m \cdot \bar{S}}{v + m}
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#555;font-size:0.8rem;margin:0.2rem 0 0.5rem 0;">'
        'v = review count for this vendor, m = Bayesian threshold, '
        'S-bar = global mean score across all vendors.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Relative rating ----
    sec("Relative Star Rating (Recommendation Tab)")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'In the Recommendation tab, ratings are relative to the uploaded vendor set. '
        'The best vendor always receives 5.0 stars and the worst always receives 1.0 star. '
        'All others are linearly interpolated between those two extremes.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \text{star}_i = 1 + \frac{S_i - S_{\min}}{S_{\max} - S_{\min}} \times 4
    \quad \in [1.0,\ 5.0]
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#555;font-size:0.8rem;margin:0.2rem 0 0.5rem 0;">'
        'S_i = raw weighted score for vendor i, '
        'S_min = lowest raw score in the set, '
        'S_max = highest raw score in the set. '
        'If all vendors have identical scores they all receive 3.0.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Absolute rating ----
    sec("Absolute Star Rating (Compare Vendors Tab)")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'The Bayesian score in [-1, +1] is mapped linearly to a 0 to 5 star scale:'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \text{rating} = \frac{S_{\text{bayes}} + 1}{2} \times 5
    """)

    # ---- No aspect detected ----
    sec("When No Aspects Are Detected")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;line-height:1.9;font-size:0.88rem;max-width:760px;">'
        'The ATE model requires a concrete noun or named entity in the sentence in order to assign '
        'a B-ASP or I-ASP tag. When a review contains only opinion words without a specific subject '
        'or object, no aspect spans are identified and the pipeline returns an empty result. '
        'This is expected behaviour and not a model failure. Such reviews are excluded from '
        'aggregation and counted separately in the output statistics.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Normalisation ----
    sec("Aspect Normalisation")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;line-height:1.9;font-size:0.88rem;max-width:760px;">'
        'Raw extracted terms vary widely across domains. When normalisation is enabled, terms are '
        'mapped to six generic categories before aggregation: quality, service, value, reliability, '
        'ambiance, and experience. This makes cross-domain and cross-vendor comparisons consistent. '
        'Normalisation can be toggled off in the sidebar to retain raw terms.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Model Evaluation ----
    sec("Model Evaluation Tab")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;line-height:1.9;font-size:0.88rem;max-width:760px;">'
        'The Model Evaluation tab runs the full pipeline on raw review data and reports how confident '
        'and consistent the model is. Because the data has no ground truth labels, evaluation is based '
        'on three measurable properties: prediction confidence, sentiment distribution balance, and '
        'aspect extraction rate.'
        '</div>',
        unsafe_allow_html=True,
    )

    sec("Confidence Score")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'The confidence score is the maximum softmax probability across the four classes:'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \text{confidence} = \max_{k \in \{1,2,3,4\}} \; p_k
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#555;font-size:0.82rem;margin:0.2rem 0 0.5rem 0;">'
        'A confidence close to 1.0 means the model is very certain. '
        'A confidence near 0.25 means the model is essentially guessing equally across all four classes.'
        '</div>',
        unsafe_allow_html=True,
    )

    sec("Low-Confidence Rate per Vendor")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'The fraction of predictions below the confidence threshold for a given vendor:'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \text{LCR}_v = \frac{\left|\{ p \in P_v \;:\; \text{confidence}(p) < \tau \}\right|}{|P_v|}
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#555;font-size:0.82rem;margin:0.2rem 0 0.5rem 0;">'
        'P_v is the set of all predictions for vendor v and tau is the confidence threshold. '
        'A high LCR suggests the reviews contain vocabulary the model was not trained on.'
        '</div>',
        unsafe_allow_html=True,
    )

    sec("Aspect Extraction Rate")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'The fraction of reviews that produced at least one aspect term:'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \text{AER} = \frac{\left|\{ r \in R \;:\; |\text{aspects}(r)| > 0 \}\right|}{|R|}
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#555;font-size:0.82rem;margin:0.2rem 0 0.5rem 0;">'
        'An AER below 0.5 means fewer than half the reviews contributed any aspect-level information, '
        'which weakens the reliability of the final ratings.'
        '</div>',
        unsafe_allow_html=True,
    )

    sec("Mean Weighted Score per Vendor")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;font-size:0.86rem;margin:0.4rem 0 0.5rem 0;">'
        'The mean weighted score across all predictions for a vendor:'
        '</div>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    \bar{s}_v = \frac{1}{|P_v|} \sum_{p \in P_v} s_p
    \quad \text{where} \quad
    s_p = \sum_{k=1}^{4} p_k \cdot w_k
    """)
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#555;font-size:0.82rem;margin:0.2rem 0 0.5rem 0;">'
        'w_k are the signed class weights: positive=+1, negative=-1, neutral=0, conflict=-0.5.'
        '</div>',
        unsafe_allow_html=True,
    )

    sec("What High Confidence Actually Means")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#666;line-height:1.9;font-size:0.88rem;max-width:760px;">'
        'High confidence does not guarantee the prediction is correct. It means the model is certain '
        'given what it learned during training. Low confidence is a useful signal: it tells you the '
        'model is uncertain and the prediction should be treated with less weight. '
        'The combination of high confidence and a balanced sentiment distribution across all four '
        'classes is the strongest indicator that the model is behaving reliably on a given dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- References ----
    sec("References")
    st.markdown(
        '<div style="font-family:IBM Plex Sans,sans-serif;color:#444;line-height:2;font-size:0.82rem;">'
        'Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019.<br>'
        'Pontiki et al. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. SemEval 2014.<br>'
        'Pontiki et al. (2016). SemEval-2016 Task 5: Aspect Based Sentiment Analysis. SemEval 2016.<br>'
        'Wolf et al. (2020). HuggingFace Transformers: State-of-the-art NLP. EMNLP 2020 System Demonstrations.<br>'
        'Vaswani et al. (2017). Attention is All You Need. NeurIPS 2017.'
        '</div>',
        unsafe_allow_html=True,
    )
