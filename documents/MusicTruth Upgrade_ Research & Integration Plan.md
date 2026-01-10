# MusicTruth Upgrade: Research & Integration Plan

**Datum:** 2026-01-10

**Auteur:** Sherlock (Manus AI)

## 1. Inleiding

Dit document presenteert een uitgebreid onderzoek naar open-source software, frameworks en AI-modellen die geïntegreerd kunnen worden in MusicTruth. Het doel is om MusicTruth te transformeren tot de meest geavanceerde en complete oplossing voor AI-muziekdetectie, -analyse en -rapportage. De aanbevelingen in dit document dienen als input voor de verdere ontwikkeling van het project.

De huidige versie van MusicTruth heeft al een solide basis met functionaliteiten zoals spectrale analyse, tempo stabiliteit, stereo imaging, stem-separatie via Demucs en een basis ML-model voor deepfake detectie. Dit plan bouwt daarop voort met een modulaire architectuur die schaalbaarheid, flexibiliteit en diepgaande analysemogelijkheden biedt.

## 2. Aanbevolen Architectuur

Een modulaire, multi-layered architectuur wordt aanbevolen om de diverse functionaliteiten te accommoderen. Dit zorgt voor flexibiliteit in het kiezen van tools en AI-modellen en maakt het systeem toekomstbestendig.

### Conceptuele Lagen:

1.  **Input Layer**: Verwerkt diverse audiobronnen (lokaal, URL, streaming).
2.  **Processing Layer**: Voert de kernanalyses uit (separatie, feature extractie, transcriptie).
3.  **Analysis & Detection Layer**: Past AI-modellen en algoritmes toe voor detectie en patroonherkenning.
4.  **Enrichment & Reporting Layer**: Verrijkt data met externe metadata en genereert rapportages.
5.  **AI Orchestration Layer**: Managet en coördineert de verschillende AI-modellen.

## 3. Audio Processing & Analyse

De kern van MusicTruth is de mogelijkheid om audio diepgaand te analyseren. Dit vereist een combinatie van state-of-the-art tools voor stem-separatie, feature extractie en muzikale analyse.

### 3.1. Stem Separatie (Source Separation)

Hoewel Demucs al geïntegreerd is, kan de toevoeging van andere modellen de gebruiker meer flexibiliteit en betere resultaten bieden, afhankelijk van de context.

| Tool/Library | Model(s) | Voordelen | Nadelen | Aanbeveling |
| :--- | :--- | :--- | :--- | :--- |
| **Demucs (Meta AI)** | `htdemucs_ft`, `htdemucs_6s` | State-of-the-art, 6-stem model | - | **Behouden & Upgraden** |
| **Ultimate Vocal Remover (UVR)** | MDX-Net, VR Arch | Zeer hoge kwaliteit vocale extractie | GUI-gebaseerd, minder direct integreerbaar | **Integreren via `python-audio-separator`** |
| **Spleeter (Deezer)** | 2, 4, 5-stem | Snel, lichtgewicht | Kwaliteit is minder dan Demucs | **Optioneel (voor snelle scans)** |

**Aanbeveling:** Upgrade naar de nieuwste Demucs-modellen en integreer **`python-audio-separator`** om programmatisch toegang te krijgen tot de superieure vocale isolatie van UVR's MDX-Net modellen. Dit geeft de gebruiker de keuze tussen de beste allround separatie (Demucs) en de beste vocale extractie (UVR).

### 3.2. Audio Feature Extractie

Librosa is de huidige standaard, maar voor diepgaandere musicologische analyse is **Essentia** een onmisbare toevoeging.

-   **Essentia (MTG/UPF):** Een C++ library met Python-bindings, specifiek ontworpen voor Music Information Retrieval (MIR). Het biedt een enorme collectie (>200) aan algoritmes voor spectrale, tonale, ritmische en high-level descriptors. De integratie van Essentia zal de analysemogelijkheden van MusicTruth significant vergroten.
    -   **Licentie:** Affero GPLv3 (vereist dat MusicTruth ook open-source blijft onder AGPLv3) of een commerciële licentie.

### 3.3. Muziektheoretische Analyse & Transcriptie

Om nummers te ontleden in noten, instrumenten en structuren, zijn de volgende tools essentieel:

-   **Spotify Basic Pitch:** Een lichtgewicht maar krachtige audio-naar-MIDI converter. Dit model kan polyfone muziek transcriberen en zelfs pitch bends detecteren. De output (MIDI) kan vervolgens verder geanalyseerd worden.
-   **music21:** Een Python toolkit voor computer-ondersteunde musicologie. Het kan MIDI-bestanden (gegenereerd door Basic Pitch) en andere muzieknotatie-formaten analyseren op harmonie, melodie, vorm en andere musicologische concepten. Dit is cruciaal voor het creëren van een 'digitale handtekening' van een artiest.

## 4. AI Detectie & Patroonherkenning

Dit is de kern van de 'Truth' in MusicTruth. Een multi-model aanpak is hier cruciaal voor robuustheid.

### 4.1. AI-gegenereerde Muziek Detectie

Er zijn diverse open-source modellen beschikbaar die verschillende aspecten van AI-gegenereerde audio detecteren.

| Model/Framework | Aanpak | Voordelen | Aanbeveling |
| :--- | :--- | :--- | :--- |
| **Deezer Deepfake Detector** | Autoencoder artifact detection | Robuust tegen genre bias, interpreteerbaar | **Integreren als primaire detector** |
| **AI Music Detection (lcrosvila)** | Hierarchical classifiers, Essentia & CLAP | Specifiek voor Suno/Udio, test tegen transformaties | **Integreren als secundaire detector** |
| **Hugging Face Audio Models** | Transformers (AST, Wav2Vec2) | Toegang tot state-of-the-art modellen, zero-shot classificatie | **Gebruiken voor custom model training & validatie** |

### 4.2. Artist Fingerprinting & Patroonherkenning

Het creëren van een 'digitale handtekening' van een artiest vereist het analyseren van patronen over meerdere werken. Dit is een complexe taak die een combinatie van technieken vereist:

1.  **Feature Aggregation:** Verzamel een breed scala aan features (spectrale, tonale, ritmische) van meerdere authentieke nummers van een artiest met **Essentia** en **librosa**.
2.  **Musicological Analysis:** Gebruik **music21** om harmonische progressies, melodische contouren en ritmische voorkeuren te analyseren.
3.  **Machine Learning:** Train een model (bijv. een Siamese Network of een One-Class SVM) op deze geaggregeerde data om een 'stijl-fingerprint' te creëren. Dit model kan vervolgens de waarschijnlijkheid berekenen dat een nieuw nummer bij de stijl van de artiest past.
4.  **Album Consistentie:** Analyseer de variatie van features binnen en tussen albums om hybride albums (AI en menselijke tracks) te detecteren. De huidige implementatie in MusicTruth kan hier als basis dienen en verder worden uitgebreid.

### 4.3. Audio Fingerprinting (Identificatie)

Voor bronverificatie en het identificeren van bekende nummers zijn traditionele audio fingerprinting libraries nuttig.

-   **Chromaprint / AcoustID (`pyacoustid`):** Kan een nummer identificeren door de fingerprint te matchen met de AcoustID database. Dit helpt bij het verifiëren van de metadata die door de gebruiker is opgegeven.
-   **Dejavu:** Kan een lokale database van fingerprints opbouwen. Dit is nuttig voor het detecteren van exacte duplicaten of samples binnen een grote collectie van een artiest.

## 5. Bron-integratie & Metadata

Een brede input en rijke metadata zijn cruciaal voor contextuele analyse.

### 5.1. Audiobronnen

-   **Lokale bestanden:** Blijven ondersteunen via `librosa` en `soundfile`.
-   **YouTube/SoundCloud etc.:** Blijven ondersteunen en uitbreiden met **`yt-dlp`**.
-   **Spotify/Deezer:** Integreer **`spotdl`** om nummers direct van deze platformen te kunnen downloaden (via YouTube Music als backend) op basis van een link.

### 5.2. Metadata APIs

Een gelaagde aanpak voor metadata verrijking wordt aanbevolen:

1.  **Primair (Muziek-specifiek):**
    -   **Spotify API (`spotipy`):** Biedt rijke audio features, populariteit, en gedetailleerde track/album/artiest data.
    -   **MusicBrainz (`musicbrainzngs`):** De 'Wikipedia voor muziek'. Essentieel voor relaties, credits en disambiguatie.
    -   **Discogs API (`python3-discogs-client`):** Zeer gedetailleerde release-informatie, inclusief specifieke persingen en personeel.
2.  **Secundair (Context & Achtergrond):**
    -   **Last.fm API (`pylast`):** Biedt 'similar artists', tags en luisteraar-data.
    -   **Genius API (`lyricsgenius`):** Voor songteksten, die ook geanalyseerd kunnen worden.
    -   **Wikipedia API (`wikipedia-api`):** Voor biografieën en achtergrondinformatie.
    -   **News API (`newsapi-python`):** Voor recente media-uitingen over een artiest.

## 6. Rapportage & Visualisatie

De output van MusicTruth moet flexibel en inzichtelijk zijn.

### 6.1. Rapportage Formaten

-   **Markdown/HTML:** Gebruik **Jinja2** templates voor het genereren van dynamische en goed gestructureerde HTML-rapporten. Deze kunnen direct in een webinterface getoond worden.
-   **PDF:** Gebruik **WeasyPrint** om de gegenereerde HTML/CSS om te zetten naar professioneel ogende PDF-rapporten. Dit biedt meer styling-mogelijkheden dan ReportLab.

### 6.2. Data Visualisatie

-   **Matplotlib/Seaborn:** Blijven gebruiken voor statische, publicatie-kwaliteit plots (spectrogrammen, heatmaps).
-   **Plotly:** Integreer Plotly voor het creëren van **interactieve** visualisaties in de HTML-rapporten. Gebruikers kunnen dan inzoomen op grafieken, hoveren voor details en zelfs audiofragmenten afspelen die gekoppeld zijn aan datapunten.

## 7. AI Provider Architectuur

Om flexibiliteit en controle te bieden, moet MusicTruth een multi-provider AI-architectuur implementeren.

### 7.1. Modulaire AI-Rollen

Definieer verschillende rollen voor AI-modellen, die door de gebruiker geconfigureerd kunnen worden:

-   **Analyse & Detectie AI:** Een model (of ensemble van modellen) dat de kern audio-analyse uitvoert.
-   **Achtergrond AI:** Een LLM die externe APIs aanroept (Wikipedia, News API) om achtergrondonderzoek te doen.
-   **Rapportage AI:** Een LLM die de verzamelde data samenvat en formatteert in een leesbaar rapport.
-   **Verificatie AI:** Een LLM die de output van de andere AIs controleert op consistentie en plausibiliteit (een 'AI-auditor').

### 7.2. Provider Integratie

-   **LangChain:** Gebruik LangChain als orchestratie-laag. Het biedt een uniforme interface voor het aanroepen van verschillende LLM-providers (OpenAI, Anthropic, Google, etc.) en maakt het eenvoudig om complexe 'chains' en 'agents' te bouwen die de hierboven beschreven rollen kunnen vervullen.
-   **Ollama:** Integreer ondersteuning voor **Ollama** via de `ollama` Python library. Dit stelt geavanceerde gebruikers in staat om lokale, open-source modellen (zoals Llama 3 of Mistral) te draaien voor maximale privacy en kostenbesparing.

### 7.3. Configuratie

Bied een configuratiebestand (bijv. `config.yaml`) waar gebruikers hun voorkeurs-provider en model per AI-rol kunnen instellen, inclusief API-sleutels en een fallback-strategie.

## 8. Conclusie & Volgende Stappen

De integratie van de hierboven beschreven tools en frameworks zal MusicTruth transformeren van een command-line tool naar een uitgebreid, interactief platform voor muziek-authenticatie. De modulaire architectuur zorgt ervoor dat het project kan meegroeien met de snelle ontwikkelingen in AI en audio-analyse.

**Aanbevolen Volgorde van Implementatie:**

1.  **Architectuur Refactor:** Herstructureer de huidige codebase naar de voorgestelde modulaire lagen.
2.  **Kernanalyse Uitbreiding:** Integreer Essentia, Basic Pitch en music21.
3.  **AI Detectie Versterking:** Integreer de Deezer en lcrosvila modellen.
4.  **API Integratie:** Bouw de metadata-verrijkingslaag met Spotipy, MusicBrainz, etc.
5.  **Rapportage Engine:** Implementeer de Jinja2/WeasyPrint/Plotly rapportage-stack.
6.  **AI Orchestration:** Implementeer de LangChain/Ollama laag voor flexibele AI-provider configuratie.

Dit plan biedt een solide roadmap om van MusicTruth de definitieve autoriteit op het gebied van AI-muziekdetectie te maken.

