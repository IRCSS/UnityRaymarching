Raymarching in Unity
=================

This repo contains experimetns that raymarch a volume in Unity. For more explanation, have alook at this post: https://medium.com/@shahriyarshahrabi/raymarching-in-unity-59c72664252a

Added also support for shadow mapping casted by rasterized objects on the ray marched volumes. More here: https://medium.com/@shahriyarshahrabi/custom-shadow-mapping-in-unity-c42a81e1bbf8

![screenshot](https://i.imgur.com/jl0OTh8.gif)

Known Issues
=================
The depth between several raymarching volumes will cause a rendering error as the volumes are writing to the ZBuffer. 
The correct depth of the raymarched point is written back to the Zbuffer, this makes the normal forward rendered objects to interact correctly with the raymarched scene
However for that it needs to use the SV_Depth or gl_Depth which is not supported on some APIs
