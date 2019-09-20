using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class CommandBufferSetup : MonoBehaviour {

    private CommandBuffer backFaceBuffer;
    private RenderTexture backFaceRenderTexture;
    private Camera mainCam;
    public Shader backFaceShader;

    public static List<Renderer> objectsToRenderBackFace=new List<Renderer>();
    private Material m;
    private void OnDisable()
    {
        CleanCommandBuffer();
        objectsToRenderBackFace.Clear();
    }

    void CleanCommandBuffer()
    {
        if (backFaceBuffer == null) return;
        backFaceBuffer.Clear();
        
    }
    

    void Start () {
		if(mainCam == null) mainCam = Camera.main;
        if (backFaceShader != null) m = new Material(backFaceShader);
        else Debug.LogError("no shader assigned for the backface pass");
        SetUpCommandBuffer();
    }
	

    void SetUpCommandBuffer()
    {
        if (backFaceBuffer != null) return;
        if (objectsToRenderBackFace.Count == 0) return;

        backFaceBuffer = new CommandBuffer();
        backFaceBuffer.name = "BackFacePass";
        
        backFaceRenderTexture = new RenderTexture(mainCam.pixelWidth, mainCam.pixelHeight, 16, RenderTextureFormat.RFloat);
        var rdID = new RenderTargetIdentifier(backFaceRenderTexture);
        backFaceBuffer.SetRenderTarget(rdID);
        backFaceBuffer.ClearRenderTarget(true, true, Color.clear, 1f);

        foreach(Renderer r in objectsToRenderBackFace)
        {
            if (r == null) Debug.LogError("A null reference in the back face objects list");
            backFaceBuffer.DrawRenderer(r, m);
        }
        backFaceBuffer.SetGlobalTexture("_BackFaceRender", rdID);

        mainCam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, backFaceBuffer);

    }
}
