# GTA VI Launch-Day Twitch Viewers Forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/h05meiherlianto/makefun-gta6forecast-01-agent2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/h05meiherlianto/makefun-gta6forecast-01-agent2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A quantile forecaster for the [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) GTA VI Twitch CCU bounty: it forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (19 November 2026, UTC), as five quantiles per hour (5%, 25%, 50%, 75%, 95%) for hours 2–24, anchored on hour 1's print.

## Model

The shape of a launch day is learned at run time from the case file's own comparables (past big game launches' hourly Twitch curves). For each comparable we take the log-ratio of each hour's viewers to hour 1's viewers, then:

- **Group A filter** — a comparable whose day peak exceeds its hour-1 print by more than 20× unlocked after 00:00 UTC (single simultaneous global unlock) and does not describe a rolling-midnight-local launch like GTA VI, so it is excluded.
- **Console-only median** — GTA VI is PS5/Xbox only, no PC. Console-only launches have a smaller hour 1 and grow more through the day as the US wakes; their median shape is blended 75/25 with the all-Group-A median.
- **Scale adjustment** — within the training set, an hour's log-ratio is strongly negatively correlated with log hour-1 (r ≈ −0.5 to −0.7 at late hours): small hour-1 launches grow much more. The median is shifted by the per-hour OLS slope of log-ratio on log hour-1 (capped at ±0.4).
- **Spread** — from the empirical quantiles of all Group-A log-ratios, centred on the blended median and scaled by 0.9.

## Validation

On the four practice launches (answers: SullyGnome's public hourly averages, the source the rules cite), mean pinball loss **0.193** vs the house baseline's **0.387** (per case: Wolverine 0.263, Red Dead Redemption 2 0.183, God of War Ragnarök 0.139, Spider-Man 2 0.185). Leave-one-out cross-validation on the comparables improves by a similar share.

## Run

`main.ts` (Node 22, standard library only, deterministic, no network) reads every `<id>.in` in the input directory and writes one `<id>.out` per case:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```
