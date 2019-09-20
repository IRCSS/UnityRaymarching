using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SetUpShaderGlobals : MonoBehaviour {

    public Texture Bluenoise;
    public Texture DebugTexture;
    public Color colorone, colortwo;
	// Use this for initialization
	void Start () {
        Shader.SetGlobalTexture("_BlueNoise", Bluenoise);
        Shader.SetGlobalTexture("_DebugTexture", DebugTexture);

    }

    private void Update()
    {
        Shader.SetGlobalColor("ColorOne", colorone);
        Shader.SetGlobalColor("ColorTwo", colortwo);
    }

}
