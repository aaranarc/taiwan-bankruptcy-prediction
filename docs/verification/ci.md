# CI results (GitHub Actions)

Generated from the job logs of run [42007215723](https://github.com/aaranarc/taiwan-bankruptcy-prediction/actions/runs/42007215723) on commit `462163d` (2026-11-05T10:01:28Z), conclusion **success**. Extracted by a script from `gh run view --log`; nothing typed by hand.

## lint (ruff)
```
All checks passed!
```

## test (pytest + coverage of app/ and ui/)
```
96 passed in 104.11s (0:01:44)
app/__init__.py                            0      0   100%
app/api/__init__.py                        0      0   100%
app/api/routes_auth.py                    57     11    81%
app/api/routes_predict.py                110     10    91%
app/cache/redis_cache.py                  49      2    96%
app/core/config.py                        16      1    94%
app/core/dependencies.py                  35      8    77%
app/core/exceptions.py                    10      1    90%
app/core/security.py                      26      2    92%
app/main.py                               31     11    65%
app/middleware/logging_middleware.py      17      0   100%
app/services/model_service.py            113      7    94%
app/services/risk.py                      38      2    95%
app/services/scenarios.py                 77      2    97%
app/utils/logger.py                       10      0   100%
ui/__init__.py                             0      0   100%
ui/account.py                             35     10    71%
ui/api.py                                 47      6    87%
ui/components.py                          99     12    88%
ui/inputs.py                              26      0   100%
ui/pages/history.py                       33      3    91%
ui/pages/model_card.py                    61      2    97%
ui/pages/portfolio.py                     63      7    89%
ui/pages/score.py                         53      2    96%
ui/pages/stress.py                        43      4    91%
ui/theme.py                               28      3    89%
TOTAL                                   1077    106    90%
```

## docker (build the API image, run it, wait for /meta)
```
Image size: 788MB
threshold 0.105 model cec199f7
```
