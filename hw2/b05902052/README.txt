First rearrange the whole data construction like below:

b05902052 \
  data \
    book_covers \
    business_cards \
    cd_covers \
    dvd_covers \
    museum_paintings \
    video_frames \
  local_feature.py
  color_feature.py
  texture_feature.py

And assure that the environment has installed following modules:
numpy
scikit-learn
opencv

To run the scripts, use following commands:
python local_feature.py
python color_feature.py
python texture_feature.py

