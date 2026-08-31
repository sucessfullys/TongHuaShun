# GEditBench-v2 Subtask ELO Report: Qwen_Image_Edit_2511

## Summary

| Metric | Value |
| --- | --- |
| Subtasks | 23 |
| Mean VC ELO | 936.7 |
| Mean VQ ELO | 900.6 |
| Mean IF ELO | 932.4 |
| Mean Overall ELO | 928.8 |
| Rank-1 Subtasks | 0 |
| Top-3 Subtasks | 4 |

## Rank Distribution

| Rank | Count |
| --- | --- |
| 3 | 4 |
| 4 | 5 |
| 5 | 14 |

## Qwen Image Edit 2511 By Subtask

| Subtask | Samples | VC | VQ | IF | Overall | Rank | Best Model | Best Overall | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `background_change` | 40 | 920 | 902 | 1019 | 949 | 5 | `FLUX2_klein_9b` | 1064 | -115 |
| `camera_motion` | 60 | 965 | 941 | 867 | 929 | 5 | `GPT_Image_1p5` | 1101 | -172 |
| `character_reference` | 40 | 796 | 947 | 955 | 908 | 5 | `GPT_Image_1p5` | 1093 | -185 |
| `chart_editing` | 45 | 918 | 859 | 837 | 872 | 5 | `Nano_Banana_Pro` | 1203 | -331 |
| `color_alteration` | 40 | 955 | 944 | 1037 | 978 | 4 | `Nano_Banana_Pro` | 1091 | -113 |
| `enhancement` | 60 | 994 | 817 | 864 | 905 | 5 | `GPT_Image_1p5` | 1087 | -182 |
| `hybrid` | 100 | 962 | 937 | 1028 | 977 | 4 | `GPT_Image_1p5` | 1039 | -62 |
| `in_image_text_translation` | 60 | 849 | 622 | 605 | 736 | 5 | `Nano_Banana_Pro` | 1302 | -566 |
| `line2image` | 40 | 1152 | 662 | 844 | 909 | 5 | `Nano_Banana_Pro` | 1046 | -137 |
| `material_modification` | 40 | 943 | 952 | 998 | 965 | 5 | `Nano_Banana_Pro` | 1071 | -106 |
| `motion_change` | 46 | 918 | 1015 | 1043 | 999 | 3 | `Nano_Banana_Pro` | 1033 | -34 |
| `object_reference` | 40 | 765 | 964 | 860 | 870 | 5 | `GPT_Image_1p5` | 1100 | -230 |
| `openset` | 100 | 897 | 909 | 880 | 896 | 5 | `GPT_Image_1p5` | 1072 | -176 |
| `portrait_beautification` | 62 | 931 | 914 | 1056 | 967 | 4 | `Nano_Banana_Pro` | 1042 | -75 |
| `relation_change` | 41 | 958 | 901 | 953 | 939 | 5 | `Nano_Banana_Pro` | 1092 | -153 |
| `size_adjustment` | 40 | 1047 | 882 | 983 | 970 | 3 | `GPT_Image_1p5` | 1067 | -97 |
| `style_reference` | 40 | 734 | 884 | 735 | 795 | 5 | `Nano_Banana_Pro` | 1126 | -331 |
| `style_transfer` | 51 | 1013 | 959 | 879 | 953 | 5 | `Seedream4p5` | 1031 | -78 |
| `subject_addition` | 40 | 930 | 969 | 1030 | 983 | 4 | `GPT_Image_1p5` | 1025 | -42 |
| `subject_removal` | 40 | 939 | 947 | 1029 | 969 | 3 | `Nano_Banana_Pro` | 1087 | -118 |
| `subject_replace` | 40 | 962 | 968 | 991 | 976 | 3 | `FLUX2_klein_9b` | 1055 | -79 |
| `text_editing` | 87 | 1016 | 885 | 980 | 956 | 4 | `Nano_Banana_Pro` | 1124 | -168 |
| `tone_transfer` | 48 | 979 | 934 | 972 | 961 | 5 | `Nano_Banana_Pro` | 1074 | -113 |

## Sorted By Qwen Rank

| Rank | Subtask | Overall | VC | VQ | IF | Gap to Best |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | `motion_change` | 999 | 918 | 1015 | 1043 | -34 |
| 3 | `subject_replace` | 976 | 962 | 968 | 991 | -79 |
| 3 | `size_adjustment` | 970 | 1047 | 882 | 983 | -97 |
| 3 | `subject_removal` | 969 | 939 | 947 | 1029 | -118 |
| 4 | `subject_addition` | 983 | 930 | 969 | 1030 | -42 |
| 4 | `color_alteration` | 978 | 955 | 944 | 1037 | -113 |
| 4 | `hybrid` | 977 | 962 | 937 | 1028 | -62 |
| 4 | `portrait_beautification` | 967 | 931 | 914 | 1056 | -75 |
| 4 | `text_editing` | 956 | 1016 | 885 | 980 | -168 |
| 5 | `material_modification` | 965 | 943 | 952 | 998 | -106 |
| 5 | `tone_transfer` | 961 | 979 | 934 | 972 | -113 |
| 5 | `style_transfer` | 953 | 1013 | 959 | 879 | -78 |
| 5 | `background_change` | 949 | 920 | 902 | 1019 | -115 |
| 5 | `relation_change` | 939 | 958 | 901 | 953 | -153 |
| 5 | `camera_motion` | 929 | 965 | 941 | 867 | -172 |
| 5 | `line2image` | 909 | 1152 | 662 | 844 | -137 |
| 5 | `character_reference` | 908 | 796 | 947 | 955 | -185 |
| 5 | `enhancement` | 905 | 994 | 817 | 864 | -182 |
| 5 | `openset` | 896 | 897 | 909 | 880 | -176 |
| 5 | `chart_editing` | 872 | 918 | 859 | 837 | -331 |
| 5 | `object_reference` | 870 | 765 | 964 | 860 | -230 |
| 5 | `style_reference` | 795 | 734 | 884 | 735 | -331 |
| 5 | `in_image_text_translation` | 736 | 849 | 622 | 605 | -566 |

## Column Meaning

- `VC`: Visual Consistency ELO.
- `VQ`: Visual Quality ELO.
- `IF`: Instruction Following ELO.
- `Overall`: joint ELO over VC, VQ, and IF.
- `Gap`: `FLUX2_klein_9b Overall ELO - Best Overall ELO`; `0` means this model ranks first on that subtask.

## Best-Performing Subtasks

None
