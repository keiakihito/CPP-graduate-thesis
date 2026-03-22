# 1candidate tags
calm
tense
energetic
lyrical
bright
heavy

# 2 pilot annotation (20 tracks)
Make sure variety 

# 3 tag refinement
Using music2Emo, and librossa for tag support 
Ex.

mock output 
music2Emo
track_001.mp3
predicted_moods = ["tense", "dramatic"]
valence = 4.2
arousal = 6.8

track_002.mp3
predicted_moods = ["calm", "sad", "melancholic"]
valence = 3.0
arousal = 2.5

track_003.mp3
predicted_moods = ["energetic", "epic", "dramatic"]
valence = 6.0
arousal = 8.7

librossa
track_id,title,valence,arousal,spectral_centroid,rms,tempo,dynamic_range
001,Beethoven_String_Quartet_1,4.2,6.8,1850,0.09,126,0.21
002,Chopin_Nocturne_Op9_No2,3.0,2.5,980,0.04,72,0.10
003,Tchaikovsky_1812_Overture,6.0,8.7,2400,0.18,138,0.35

# 4 final tag set
Maybe

| tag       | frequency |
| --------- | --------- |
| calm      | 12        |
| tense     | 9         |
| energetic | 7         |
| lyrical   | 14        |
| bright    | 1         |
| heavy     | 2         |

birght and heavy are not useful in thie project.

Then we refine
calm
tense
energetic
lyrical


# 5 full archive annotation

