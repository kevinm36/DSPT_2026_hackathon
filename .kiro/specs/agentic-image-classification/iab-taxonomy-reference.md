# IAB Content Taxonomy Tier 1 — Reference

## Overview

The IAB (Interactive Advertising Bureau) Content Taxonomy is the industry standard for categorizing digital content and advertising. We use **Tier 1** (~30 top-level categories) as our fixed label space for image classification.

Source: [IAB Tech Lab Content Taxonomy](https://iabtechlab.com/standards/content-taxonomy/)

## Full Tier 1 Category List

| Index | Category Name | Description | ADS-16 Relevance |
|-------|--------------|-------------|------------------|
| 1 | Arts & Entertainment | Movies, music, TV, celebrities, visual arts | High — media ads |
| 2 | Automotive | Cars, trucks, motorcycles, parts, services | High — direct category match (Cat1) |
| 3 | Business & Finance | Business news, industries, markets | Medium |
| 4 | Careers | Job search, resume, workplace | Low |
| 5 | Education | Schools, learning, online courses | Low |
| 6 | Family & Parenting | Pregnancy, babies, children, family life | High — Baby Products (Cat2) |
| 7 | Food & Drink | Recipes, restaurants, beverages, cooking | High — Grocery (Cat9) |
| 8 | Health & Fitness | Medicine, wellness, exercise, nutrition | High — Health & Beauty (Cat3) |
| 9 | Hobbies & Interests | Crafts, collecting, games, outdoor activities | Medium |
| 10 | Home & Garden | Interior design, gardening, home improvement | High — Garden & Outdoor (Cat8), Kitchen & Home (Cat10) |
| 11 | Law, Government & Politics | Legal, government, political news | Low |
| 12 | News & Current Events | Breaking news, local/national/world news | Low |
| 13 | Personal Finance | Banking, investing, insurance, credit | Low |
| 14 | Pets | Dogs, cats, pet care, veterinary | High — Pet Supplies (Cat15) |
| 15 | Real Estate | Buying, selling, renting homes | Low |
| 16 | Religion & Spirituality | Faith, worship, spiritual practices | Low |
| 17 | Science | Research, discoveries, space, environment | Low |
| 18 | Shopping | Deals, coupons, product reviews, retail | High — general commerce |
| 19 | Sports | Professional/amateur sports, fitness | High — Sports & Outdoors (Cat17) |
| 20 | Style & Fashion | Clothing, accessories, beauty, trends | High — Clothing & Shoes (Cat0) |
| 21 | Technology & Computing | Gadgets, software, internet, mobile | High — Consumer Electronics (Cat5), Software (Cat16) |
| 22 | Travel | Destinations, hotels, flights, tourism | Medium |
| 23 | Gaming | Video games, PC games, mobile games | High — Console & Video Games (Cat6) |
| 24 | Music & Audio | Music streaming, instruments, audio equipment | High — Musical Instruments (Cat13) |
| 25 | Television | TV shows, streaming, broadcast | Medium |
| 26 | Dating | Online dating, relationships | High — Dating Sites (Cat19) |
| 27 | Gambling | Betting, casinos, lottery | High — Betting (Cat11) |
| 28 | Jewelry & Luxury | Fine jewelry, watches, luxury goods | High — Jewellery & Watches (Cat12) |
| 29 | Office & Professional | Office supplies, business tools | High — Office Products (Cat14) |
| 30 | DIY & Home Improvement | Tools, hardware, home repair | High — DIY & Tools (Cat7) |

## Mapping to ADS-16 Product Categories

The ADS-16 dataset uses 20 product/service categories. Here's how they map to IAB Tier 1:

| ADS-16 Category (Cat Index) | Primary IAB Tier 1 | Secondary IAB Tier 1 |
|------------------------------|-------------------|---------------------|
| Clothing & Shoes (0) | Style & Fashion | Shopping |
| Automotive (1) | Automotive | — |
| Baby Products (2) | Family & Parenting | Shopping |
| Health & Beauty (3) | Health & Fitness | Style & Fashion |
| Media - BMVD (4) | Arts & Entertainment | Music & Audio |
| Consumer Electronics (5) | Technology & Computing | Shopping |
| Console & Video Games (6) | Gaming | Technology & Computing |
| DIY & Tools (7) | DIY & Home Improvement | Home & Garden |
| Garden & Outdoor Living (8) | Home & Garden | Hobbies & Interests |
| Grocery (9) | Food & Drink | Shopping |
| Kitchen & Home (10) | Home & Garden | Shopping |
| Betting (11) | Gambling | — |
| Jewellery & Watches (12) | Jewelry & Luxury | Style & Fashion |
| Musical Instruments (13) | Music & Audio | Hobbies & Interests |
| Office Products (14) | Office & Professional | Business & Finance |
| Pet Supplies (15) | Pets | Shopping |
| Computer Software (16) | Technology & Computing | — |
| Sports & Outdoors (17) | Sports | Health & Fitness |
| Toys & Games (18) | Family & Parenting | Gaming |
| Dating Sites (19) | Dating | — |

## Classification Prompt Template

The following prompt template is used when submitting images to the vision-language model:

```
You are classifying advertisement images into IAB Content Taxonomy Tier 1 categories.

Given the image, identify which of the following categories apply. An image may belong to multiple categories. Return ONLY category names from this exact list:

[Full list of 30 categories]

Rules:
- Return 1-5 categories per image
- Use exact category names as listed above
- If unsure, prefer the most specific applicable category
- Return results as a JSON array of strings

Example output: ["Style & Fashion", "Shopping"]
```

## Validation Rules

1. **Exact match required**: Category names must match the canonical list exactly (case-sensitive after normalization)
2. **Minimum 1 category**: Every image must be classified into at least one category
3. **Soft maximum 5 categories**: More than 5 triggers a warning but is not rejected
4. **No duplicates**: Each category appears at most once per image
5. **Known aliases**: Common model hallucinations and their corrections:
   - "Fashion" → "Style & Fashion"
   - "Tech" / "Technology" → "Technology & Computing"
   - "Food" → "Food & Drink"
   - "Finance" → "Personal Finance" or "Business & Finance" (context-dependent)
   - "Games" / "Video Games" → "Gaming"

## Notes

- The IAB taxonomy has Tier 2 (~400 categories) and Tier 3 (~700+ categories) for future granularity expansion
- Starting with Tier 1 keeps the feature space manageable for LR and reduces sparsity
- If Tier 1 proves too coarse, the pipeline can be re-run with Tier 2 categories without architectural changes
