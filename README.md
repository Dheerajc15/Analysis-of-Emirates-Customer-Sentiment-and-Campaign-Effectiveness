# 🛠️ Project Structure & Methodology

---

## 📍 Phase 1: Sentiment Analysis & Topic Modeling

### 🎯 Objective  
Identify the “why” behind customer reviews and uncover key experience drivers.

### 📚 Libraries Used
- **pandas**, **numpy**
- **nltk** (tokenization, lemmatization, stopword removal)
- **scikit-learn** (TfidfVectorizer, LDA)
- **vaderSentiment**

### 🔍 Methodology

#### **1. Sentiment Analysis**
- Applied **VADER** to thousands of Kaggle reviews.  
- Computed **compound sentiment scores** to classify review polarity.

#### **2. Text Preprocessing**
Performed complete NLP cleaning:
- Tokenization  
- Lemmatization  
- Stopword removal  

#### **3. Topic Modeling**
- Vectorized text using **TF-IDF**.  
- Applied **LDA** to extract recurring themes from:  
  - ⭐ 1-star reviews (pain points)  
  - ⭐ 5-star reviews (positive experiences)

### 🧾 Key Outcomes

#### ⭐ Praise Points  
- Cabin Crew Service  
- Inflight Entertainment (ICE)  
- A380 Experience  

#### ⚠️ Pain Points  
- Call Center Customer Service  
- Refund & Claims Processing  
- Baggage Handling  


---

## 📍 Phase 2: Competitive Benchmarking

### 🎯 Objective  
Compare Emirates against Qatar Airways and Etihad to evaluate competitive positioning.

### 📚 Libraries Used
- **seaborn**, **matplotlib**
- **pandas**

### 🔍 Methodology

#### **1. Sentiment Distribution**
- Used **Seaborn boxplots** to compare sentiment score variance across the three airlines.

#### **2. Rating Comparison**
- Built **grouped bar charts** for structured review metrics such as:  
  - Seat Comfort  
  - Value for Money  
  - Food & Beverages  

#### **3. Time-Series Sentiment Tracking**
- Plotted **average sentiment over time** to visualize brand perception trends.

### 🧾 Key Outcome  
A clear comparison showing:

#### 🌟 Strengths (Emirates)
- Entertainment quality  
- Cabin Crew consistency  

#### ⚠️ Areas to Improve
- Value for Money relative to Qatar Airways  
- Responsiveness in customer support  


---

## 📍 Phase 3: Campaign Effectiveness & Live Data Integration

### 🎯 Objective  
Measure the ROI and business impact of Emirates’ marketing efforts and global sponsorships.

### 📚 Libraries Used
- **pytrends** (Google Trends API)  
- **requests**  
- **BeautifulSoup4** (Skytrax scraping)  
- **NewsAPI**  
- **pandas**

### 🔍 Methodology

#### **1. Event Data Extraction**
- Extracted campaign and sponsorship timelines from the **Emirates 2024–25 Annual Report**.

#### **2. Public Interest Tracking**
- Collected **12 months of Google search interest** via pytrends.

#### **3. Live Sentiment Data**
- Scraped **Skytrax customer reviews** using BeautifulSoup4.  
- Applied VADER to compute **real-time sentiment trends**.

#### **4. News Sentiment Monitoring**
- Retrieved latest 100 Emirates-related news articles using NewsAPI.  
- Scored sentiment to assess media tone.

### 🧾 Key Outcome  
A master correlation plot revealed:

- Significant **spikes in Google search interest** following major sponsorship announcements.  
- **Flat comparative trends** for Qatar and Etihad, showing Emirates’ stronger brand pull.  
- Strong evidence that sponsorship activities directly **boost brand buzz** and public interest.


---

## ⚙️ Core Technologies & Libraries

### 🔢 Data Manipulation
- pandas  
- numpy  

### 🧠 NLP & Machine Learning
- nltk  
- scikit-learn (LDA, TfidfVectorizer)  
- vaderSentiment  

### 📊 Data Visualization
- matplotlib  
- seaborn  

### 🌐 Data Collection
- pytrends (Google Trends)  
- requests  
- BeautifulSoup4 (Skytrax scraping)  
- NewsAPI (news sentiment extraction)  

---
