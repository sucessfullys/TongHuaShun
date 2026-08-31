# FLUX2_klein_9b Subtask ELO Table

## Summary

- Source HTML: `tmp_elo_subtask_flux2_klein_9b_table.html`
- Model: `FLUX2_klein_9b`
- Subtasks covered: **23**
- Mean VC / VQ / IF ELO: **1046.5 / 933.8 / 1002.0**
- Mean Overall ELO: **991.7**
- Rank-1 subtasks: **2**
- Top-3 subtasks: **4**
- Best-performing subtasks: `background_change`, `subject_replace`

## Rank Distribution

| Rank | Count |
| ---: | ---: |
| 1 | 2 |
| 2 | 2 |
| 4 | 4 |
| 5 | 5 |
| 6 | 4 |
| 7 | 5 |
| 8 | 1 |

## FLUX2 Klein 9B By Subtask

| Subtask | Samples | VC | VQ | IF | Overall | Rank | Best Model | Best Overall | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `background_change` | 40 | 1073 | 989 | 1212 | 1080 | 1 | `FLUX2_klein_9b` | 1080 | 0 |
| `camera_motion` | 60 | 977 | 879 | 1049 | 970 | 7 | `GPT_Image_1p5` | 1088 | -118 |
| `character_reference` | 40 | 765 | 891 | 1080 | 926 | 8 | `GPT_Image_1p5` | 1086 | -160 |
| `chart_editing` | 45 | 1078 | 985 | 916 | 995 | 4 | `Nano_Banana_Pro` | 1276 | -281 |
| `color_alteration` | 40 | 1052 | 890 | 942 | 959 | 7 | `Nano_Banana_Pro` | 1113 | -154 |
| `enhancement` | 60 | 976 | 1020 | 1009 | 999 | 5 | `Nano_Banana_Pro` | 1080 | -81 |
| `hybrid` | 100 | 1015 | 874 | 962 | 953 | 7 | `Nano_Banana_Pro` | 1041 | -88 |
| `in_image_text_translation` | 60 | 963 | 725 | 997 | 914 | 6 | `Nano_Banana_Pro` | 1277 | -363 |
| `line2image` | 40 | 826 | 1015 | 1092 | 985 | 5 | `Nano_Banana_Pro` | 1057 | -72 |
| `material_modification` | 40 | 1110 | 938 | 944 | 994 | 5 | `Nano_Banana_Pro` | 1114 | -120 |
| `motion_change` | 46 | 1153 | 896 | 969 | 999 | 5 | `Nano_Banana_Pro` | 1055 | -56 |
| `object_reference` | 40 | 990 | 955 | 1007 | 982 | 6 | `GPT_Image_1p5` | 1079 | -97 |
| `openset` | 100 | 967 | 948 | 1011 | 974 | 5 | `GPT_Image_1p5` | 1100 | -126 |
| `portrait_beautification` | 62 | 1191 | 941 | 992 | 1029 | 4 | `FireRed_Image_Edit` | 1063 | -34 |
| `relation_change` | 41 | 1137 | 942 | 833 | 973 | 7 | `Nano_Banana_Pro` | 1081 | -108 |
| `size_adjustment` | 40 | 1199 | 955 | 869 | 1003 | 4 | `Nano_Banana_Pro` | 1068 | -65 |
| `style_reference` | 40 | 965 | 988 | 978 | 976 | 4 | `Nano_Banana_Pro` | 1160 | -184 |
| `style_transfer` | 51 | 869 | 1019 | 1079 | 993 | 6 | `Seedream4p5` | 1068 | -75 |
| `subject_addition` | 40 | 1258 | 891 | 963 | 1020 | 2 | `Nano_Banana_Pro` | 1058 | -38 |
| `subject_removal` | 40 | 1231 | 994 | 1092 | 1087 | 2 | `Nano_Banana_Pro` | 1108 | -21 |
| `subject_replace` | 40 | 1243 | 924 | 1057 | 1057 | 1 | `FLUX2_klein_9b` | 1057 | 0 |
| `text_editing` | 87 | 1109 | 889 | 881 | 955 | 7 | `Nano_Banana_Pro` | 1118 | -163 |
| `tone_transfer` | 48 | 922 | 929 | 1112 | 987 | 6 | `Nano_Banana_Pro` | 1082 | -95 |

## Sorted By FLUX Rank

| Rank | Subtask | Overall | VC | VQ | IF | Gap to Best |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `background_change` | 1080 | 1073 | 989 | 1212 | 0 |
| 1 | `subject_replace` | 1057 | 1243 | 924 | 1057 | 0 |
| 2 | `subject_removal` | 1087 | 1231 | 994 | 1092 | -21 |
| 2 | `subject_addition` | 1020 | 1258 | 891 | 963 | -38 |
| 4 | `portrait_beautification` | 1029 | 1191 | 941 | 992 | -34 |
| 4 | `size_adjustment` | 1003 | 1199 | 955 | 869 | -65 |
| 4 | `chart_editing` | 995 | 1078 | 985 | 916 | -281 |
| 4 | `style_reference` | 976 | 965 | 988 | 978 | -184 |
| 5 | `enhancement` | 999 | 976 | 1020 | 1009 | -81 |
| 5 | `motion_change` | 999 | 1153 | 896 | 969 | -56 |
| 5 | `material_modification` | 994 | 1110 | 938 | 944 | -120 |
| 5 | `line2image` | 985 | 826 | 1015 | 1092 | -72 |
| 5 | `openset` | 974 | 967 | 948 | 1011 | -126 |
| 6 | `style_transfer` | 993 | 869 | 1019 | 1079 | -75 |
| 6 | `tone_transfer` | 987 | 922 | 929 | 1112 | -95 |
| 6 | `object_reference` | 982 | 990 | 955 | 1007 | -97 |
| 6 | `in_image_text_translation` | 914 | 963 | 725 | 997 | -363 |
| 7 | `relation_change` | 973 | 1137 | 942 | 833 | -108 |
| 7 | `camera_motion` | 970 | 977 | 879 | 1049 | -118 |
| 7 | `color_alteration` | 959 | 1052 | 890 | 942 | -154 |
| 7 | `text_editing` | 955 | 1109 | 889 | 881 | -163 |
| 7 | `hybrid` | 953 | 1015 | 874 | 962 | -88 |
| 8 | `character_reference` | 926 | 765 | 891 | 1080 | -160 |

## Column Meaning

- `VC`: Visual Consistency ELO.
- `VQ`: Visual Quality ELO.
- `IF`: Instruction Following ELO.
- `Overall`: joint ELO over VC, VQ, and IF.
- `Gap`: `FLUX2_klein_9b Overall ELO - Best Overall ELO`; `0` means this model ranks first on that subtask.
