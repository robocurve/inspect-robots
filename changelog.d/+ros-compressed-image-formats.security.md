**ROS plugin:** `CompressedImage` payloads are decoded only as JPEG or PNG,
the two formats the plugin documents. Pillow previously chose a decoder from
the payload's magic bytes, so a rosbridge peer could reach any decoder Pillow
ships, including the EPS one, which runs Ghostscript on the payload. Other
formats now raise the usual `ValueError`.
