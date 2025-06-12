| Feature | spatial | object | goal | 10 | lm_90 |
| --- | --- | --- | --- | --- | --- |
| action | Tensor: ['7'] float32 | Tensor: ['7'] float32 | Tensor: ['7'] float32 | Tensor: ['7'] float32 | Tensor: ['7'] float32 |
| discount | Tensor: N/A float32 | Tensor: N/A float32 | Tensor: N/A float32 | Tensor: N/A float32 | Tensor: N/A float32 |
| is_first | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool |
| is_last | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool |
| is_terminal | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool | Tensor: N/A bool |
| language_instruction | Text | Text | Text | Text | Text |
| language_motions | N/A | N/A | N/A | N/A | Text |
| language_motions_future | N/A | N/A | N/A | N/A | Text |
| observation.image | Image: ['256', '256', '3'] uint8 | Image: ['256', '256', '3'] uint8 | Image: ['256', '256', '3'] uint8 | Image: ['256', '256', '3'] uint8 | Image: ['224', '224', '3'] uint8 |
| observation.joint_state | Tensor: ['7'] float32 | Tensor: ['7'] float32 | Tensor: ['7'] float32 | Tensor: ['7'] float32 | Tensor: ['7'] float32 |
| observation.seg | N/A | N/A | N/A | N/A | Image: ['224', '224', '1'] uint8 |
| observation.state | Tensor: ['8'] float32 | Tensor: ['8'] float32 | Tensor: ['8'] float32 | Tensor: ['8'] float32 | Tensor: ['8'] float32 |
| observation.wrist_image | Image: ['256', '256', '3'] uint8 | Image: ['256', '256', '3'] uint8 | Image: ['256', '256', '3'] uint8 | Image: ['256', '256', '3'] uint8 | Image: ['224', '224', '3'] uint8 |
| reward | Tensor: N/A float32 | Tensor: N/A float32 | Tensor: N/A float32 | Tensor: N/A float32 | Tensor: N/A float32 |

## Language Instruction Statistics
| Dataset | Processed Episodes | Skipped Records | Unique Instructions |
|---|---|---|---|
| spatial | 432 | 0 | 10 |
| object | 454 | 0 | 10 |
| goal | 428 | 0 | 10 |
| 10 | 379 | 0 | 10 |
| lm_90 | 3911 | 6 | 73 |

### spatial Details

| Instruction | Episode Count |
|---|---|
| `pick up the black bowl next to the plate and place it on the plate` | 47 |
| `pick up the black bowl next to the cookie box and place it on the plate` | 46 |
| `pick up the black bowl from table center and place it on the plate` | 46 |
| `pick up the black bowl between the plate and the ramekin and place it on the plate` | 45 |
| `pick up the black bowl next to the ramekin and place it on the plate` | 45 |
| `pick up the black bowl on the wooden cabinet and place it on the plate` | 44 |
| `pick up the black bowl on the cookie box and place it on the plate` | 43 |
| `pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate` | 42 |
| `pick up the black bowl on the ramekin and place it on the plate` | 39 |
| `pick up the black bowl on the stove and place it on the plate` | 35 |

### object Details

| Instruction | Episode Count |
|---|---|
| `pick up the chocolate pudding and place it in the basket` | 50 |
| `pick up the salad dressing and place it in the basket` | 47 |
| `pick up the bbq sauce and place it in the basket` | 46 |
| `pick up the orange juice and place it in the basket` | 45 |
| `pick up the milk and place it in the basket` | 45 |
| `pick up the ketchup and place it in the basket` | 45 |
| `pick up the cream cheese and place it in the basket` | 45 |
| `pick up the butter and place it in the basket` | 45 |
| `pick up the alphabet soup and place it in the basket` | 44 |
| `pick up the tomato sauce and place it in the basket` | 42 |

### goal Details

| Instruction | Episode Count |
|---|---|
| `turn on the stove` | 50 |
| `put the bowl on the plate` | 49 |
| `put the bowl on the stove` | 48 |
| `put the wine bottle on top of the cabinet` | 47 |
| `put the bowl on top of the cabinet` | 46 |
| `open the middle drawer of the cabinet` | 43 |
| `put the cream cheese in the bowl` | 40 |
| `put the wine bottle on the rack` | 36 |
| `open the top drawer and put the bowl inside` | 36 |
| `push the plate to the front of the stove` | 33 |

### 10 Details

| Instruction | Episode Count |
|---|---|
| `put both the cream cheese box and the butter in the basket` | 49 |
| `put both the alphabet soup and the cream cheese box in the basket` | 43 |
| `turn on the stove and put the moka pot on it` | 41 |
| `pick up the book and place it in the back compartment of the caddy` | 41 |
| `put the white mug on the left plate and put the yellow and white mug on the right plate` | 38 |
| `put the white mug on the plate and put the chocolate pudding to the right of the plate` | 36 |
| `put the black bowl in the bottom drawer of the cabinet and close it` | 35 |
| `put the yellow and white mug in the microwave and close it` | 34 |
| `put both the alphabet soup and the tomato sauce in the basket` | 33 |
| `put both moka pots on the stove` | 29 |

### lm_90 Details

| Instruction | Episode Count |
|---|---|
| `put the black bowl on top of the cabinet` | 146 |
| `pick up the book and place it in the left compartment of the caddy` | 137 |
| `put the black bowl in the top drawer of the cabinet` | 100 |
| `turn on the stove` | 99 |
| `close the top drawer of the cabinet` | 99 |
| `pick up the book and place it in the right compartment of the caddy` | 97 |
| `pick up the tomato sauce and put it in the basket` | 94 |
| `open the top drawer of the cabinet` | 92 |
| `put the black bowl on the plate` | 91 |
| `pick up the book and place it in the front compartment of the caddy` | 91 |
| `pick up the alphabet soup and put it in the basket` | 85 |
| `turn on the stove and put the frying pan on it` | 73 |
| `pick up the butter and put it in the tray` | 50 |
| `put the chocolate pudding to the right of the plate` | 50 |
| `close the microwave` | 49 |
| `put the butter at the back in the top drawer of the cabinet and close it` | 49 |
| `stack the black bowl at the front on the black bowl in the middle` | 49 |
| `put the butter at the front in the top drawer of the cabinet and close it` | 49 |
| `pick up the chocolate pudding and put it in the tray` | 49 |
| `pick up the black bowl on the left and put it in the tray` | 49 |
| `put the middle black bowl on the plate` | 49 |
| `pick up the cream cheese and put it in the tray` | 49 |
| `put the middle black bowl on top of the cabinet` | 48 |
| `put the chocolate pudding to the left of the plate` | 48 |
| `pick up the book on the right and place it on the cabinet shelf` | 47 |
| `pick up the book on the right and place it under the cabinet shelf` | 47 |
| `put the frying pan on the stove` | 47 |
| `put the yellow and white mug on the right plate` | 47 |
| `pick up the book in the middle and place it on the cabinet shelf` | 47 |
| `pick up the tomato sauce and put it in the tray` | 47 |
| `pick up the ketchup and put it in the tray` | 47 |
| `pick up the yellow and white mug and place it to the right of the caddy` | 46 |
| `pick up the salad dressing and put it in the tray` | 46 |
| `pick up the white mug and place it to the right of the caddy` | 46 |
| `pick up the red mug and place it to the right of the caddy` | 46 |
| `open the microwave` | 46 |
| `pick up the cream cheese box and put it in the basket` | 46 |
| `put the black bowl at the back on the plate` | 46 |
| `put the white bowl to the right of the plate` | 46 |
| `open the bottom drawer of the cabinet` | 46 |
| `close the top drawer of the cabinet and put the black bowl on top of it` | 46 |
| `put the black bowl in the bottom drawer of the cabinet` | 46 |
| `put the frying pan under the cabinet shelf` | 45 |
| `pick up the ketchup and put it in the basket` | 45 |
| `pick up the book and place it in the back compartment of the caddy` | 45 |
| `put the red mug on the right plate` | 45 |
| `put the yellow and white mug to the front of the white mug` | 44 |
| `put the chocolate pudding in the top drawer of the cabinet and close it` | 44 |
| `pick up the alphabet soup and put it in the tray` | 44 |
| `put the red mug on the plate` | 44 |
| `stack the left bowl on the right bowl and place them in the tray` | 44 |
| `put the moka pot on the stove` | 44 |
| `pick up the book on the left and place it on top of the shelf` | 44 |
| `put the white bowl on top of the cabinet` | 44 |
| `put the white mug on the left plate` | 44 |
| `put the wine bottle in the bottom drawer of the cabinet` | 43 |
| `put the white mug on the plate` | 43 |
| `turn off the stove` | 42 |
| `put the white bowl on the plate` | 42 |
| `put the black bowl at the front on the plate` | 41 |
| `pick up the milk and put it in the basket` | 41 |
| `put the right moka pot on the stove` | 41 |
| `close the bottom drawer of the cabinet` | 41 |
| `put the ketchup in the top drawer of the cabinet` | 41 |
| `stack the middle black bowl on the back black bowl` | 39 |
| `pick up the orange juice and put it in the basket` | 39 |
| `open the top drawer of the cabinet and put the bowl in it` | 38 |
| `put the wine bottle on the wine rack` | 38 |
| `put the frying pan on the cabinet shelf` | 38 |
| `put the frying pan on top of the cabinet` | 37 |
| `stack the right bowl on the left bowl and place them in the tray` | 36 |
| `close the bottom drawer of the cabinet and open the top drawer` | 34 |
| `put the red mug on the left plate` | 34 |

### Missing Instructions from LIBERO-90

The following instructions from the complete LIBERO-90 task list are not present in the processed `lm_90` dataset above:

- `put the butter at the back in the top drawer of the cabinet`
- `pick up the butter and put it in the basket`

Additionally, the following instruction is present in `lm_90` but not in the canonical LIBERO-90 list:

- `put the butter at the back in the top drawer of the cabinet and close it`