# MusicTruth .env Configuration Guide

## Quick Setup

1. **Copy the template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** with your API keys:
   ```bash
   notepad .env  # Windows
   nano .env     # Linux/Mac
   ```

3. **Fill in your credentials** (see sections below)

---

## API Keys

### Google Gemini (Recommended)
Get your free API key: https://aistudio.google.com/app/apikey

```env
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.0-flash-exp
```

### OpenAI
Get your API key: https://platform.openai.com/api-keys

```env
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4
```

### Anthropic Claude
Get your API key: https://console.anthropic.com/

```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

---

## Metadata APIs (Optional)

### Spotify
Create an app: https://developer.spotify.com/dashboard

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

### MusicBrainz
No API key required, but you can set a contact email:

```env
MUSICBRAINZ_CONTACT_EMAIL=your_email@example.com
```

---

## Default Settings

```env
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_ANALYSIS_MODE=standard
DEFAULT_OUTPUT_FORMATS=json,html
```

---

## Security Notes

- ⚠️ **Never commit `.env` to Git!** (It's already in `.gitignore`)
- 🔒 Keep your API keys private
- 🔄 Rotate keys if accidentally exposed
