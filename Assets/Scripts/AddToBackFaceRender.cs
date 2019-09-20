using System.Collections;
using System.Collections.Generic;
using UnityEngine;

 [RequireComponent(typeof(Renderer))]
public class AddToBackFaceRender : MonoBehaviour {
    public Renderer r;
    private void OnEnable()
    {
        if (r == null) r =this.GetComponent<Renderer>();
        if(!CommandBufferSetup.objectsToRenderBackFace.Contains(r)) CommandBufferSetup.objectsToRenderBackFace.Add(r) ;
    }

    private void OnDisable()
    {
        if (r == null) r = this.GetComponent<Renderer>();
        if (CommandBufferSetup.objectsToRenderBackFace.Contains(r)) CommandBufferSetup.objectsToRenderBackFace.Remove(r);
    }
}
