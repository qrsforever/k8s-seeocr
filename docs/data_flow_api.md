
```

COS_DOMAIN: htpps://cosftpd-1301930378.cos.ap-beijing.myqcloud.com

    ╭───────────╮
    │           │                  ${COS_DOMAIN}/holyhand/008e9b61ac6c/20221104/images/P22110420294710.jpg
    │  Camera   │                                        ╭─────────────╮
    │    +ftp   │                              put       │             │
    │           │                       ───────────────> │    COS      │
    ╰───────────╯                      ╱                 │             │
          │                           ╱                  ╰─────────────╯
          │                          ╱                            ╭─────────────────────────────────────────────────────╮
          ╰──╮                      ╱                             │{                                                    │
             │                     ╱                              │   "errno": 0,                                       │
             v                    ╱                               │   "pigeon": { "msgkey": "seeocr_output"},           │
       ╔══════════════════╗      ╱                                |   "ocr_res_json": "${COS_DOMAIN}/xxx/result.json"   │
       ║     CosFtpd      ║     ╱                                 │}                                                    │
       ║            ╭─────╨──────╮                                ╰─────────────────────────────────────────────────────╯
       ║            │ w  fifo  r │                                                 result
       ║            ╰─────╥──────╯                               ╭──────────────╮                ╭────────────────╮
       ╚══════════════════╝    │                                 │              │<───────────────│                │
                               │                            ╭───>│   Kafka      │                │    seeocr      │
                               │          ╭───────────╮     │    │              │───────────────>│                │
                               │ post     │           │─────╯    ╰──────────────╯                ╰────────────────╯
                               ╰────────> │  业务SQL  │                ^          seeocr_input
                                http      │           │                │
                                          ╰───────────╯                ╰──────────────────────────────────╮
                        /cosftpd/on_upload                                                                │
 _______________________________________________________________________________________________          │
/\                                                                                              \         │
\_| {                                                                                           |         │
  |     "ftp_user":"holyhand",                                                                  |         │
  |     "mac":"008e9b61ac6c",                                                                   |         │
  |     "cos_url":"${COS_DOMAIN}/holyhand/008e9b61ac6c/20221104/images/P22110420294710.jpg"     |         │
  | }                                                                                           |         │
  |   __________________________________________________________________________________________|_        │
   \_/____________________________________________________________________________________________/       │
                                                                                                          │
                _______________________________________________________________________________________________
               /\                                                                                              \
               \_|                                                                                             |
                 | {                                                                                           |
                 |     "cfg":{                                                                                 |
                 |         "pigeon":{                                                                          |
                 |             "msgkey":"seeocr_output"                                                        |
                 |         },                                                                                  |
                 |         "source":"${COS_DOMAIN}/holyhand/008e9b61ac6c/20221104/images/P22110420294710.jpg", |
                 |         "devmode": false,                                                                   |
                 |         "ocr_keys":[ "产品型号", "生产人数"],                                               |
                 |         "det_max_side_len":800,                                                             |
                 |         "det_thresh":0.3,                                                                   |
                 |         "det_box_trhesh":0.6,                                                               |
                 |         "det_unclip_ratio":1.5,                                                             |
                 |         "rec_batch_num":6,                                                                  |
                 |         "rec_image_shape":[3, 48, 320],                                                     |
                 |         "process_code":""                                                                   |
                 |     }                                                                                       |
                 | }                                                                                           |
                 |   __________________________________________________________________________________________|_
                  \_/____________________________________________________________________________________________/

```
