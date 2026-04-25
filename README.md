# ai-exposure-asset-pricing

Empirical study of AI exposure as a risk factor in equity markets using patents, earnings call transcripts, and asset pricing models.

---

## AI Patent Data Construction

### Data Source

The patent-based AI exposure measure is constructed using the **USPTO Artificial Intelligence Patent Dataset (AIPD)**.  
This dataset provides patent-level predictions of whether a patent is related to artificial intelligence, based on machine learning and NLP models applied to patent text.

Each observation corresponds to a patent document and includes:
- Patent identifier (`doc_id`)
- Publication date (`pub_dt`)
- AI classification probabilities (e.g., `predict86_any_ai`)

In this project, we use the `predict86_any_ai` indicator to identify AI-related patents, as it provides a stricter and higher-precision classification.

---

### Data Availability

The raw patent dataset is **not included in this repository** due to its large size.

Users who wish to replicate the analysis can download the dataset directly from the USPTO and PatentsView:

- USPTO Artificial Intelligence Patent Dataset (AIPD)
- PatentsView Disambiguated Assignee Dataset, specifically `g_assignee_disambiguated.tsv`

Due to computational and storage constraints, this repository includes only the intermediate and processed data necessary to reproduce key results.

---

### Intermediate Data

To facilitate reproducibility, we include a cleaned mapping file used to link patent assignee names to public firm tickers: `data/intermediate/top_500_ai_firms_with_ticker.xlsx`


This file contains:
- Standardized company names from the patent data
- Corresponding stock tickers for U.S.-listed parent companies

This mapping ensures accurate merging with financial datasets and S&P 500 constituents.

---

### Data Processing Pipeline

The patent data processing follows these steps:

#### 1. Filter patents
- Keep granted patents (`flag_patent == 1`)
- Restrict to recent years (2016-2023)

#### 2. Identify AI patents
- Use `predict86_any_ai` as the AI classification indicator

#### 3. Map patents to firms
The AIPD dataset is defined at the patent level and does not directly include firm identifiers.  
To construct firm-level measures, we merge the AIPD data with the **PatentsView disambiguated assignee dataset**, which provides standardized firm ownership information.

#### 4. Aggregate to firm level
- Count total patents and AI patents by firm and time period (quarter)

#### 5. Match to public firms
- Merge with the ticker mapping file: `data/intermediate/top_500_ai_firms_with_ticker.xlsx`

#### 6. Construct AI exposure measures
- AI patent count  
- Total patent count  
- AI share = AI patents / total patents  

---

### Output

The final output of this step is a firm-quarter level dataset: `data/processed/firm_quarter_with_ticker_only.csv`


This dataset is used in subsequent empirical analysis, including asset pricing regressions.

---

### Code

The patent data processing is implemented in: `Thesis_Patent_Data_Ticker_Matching.ipynb`


Please update local file paths before running the notebook.


