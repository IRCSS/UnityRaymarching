using System.Collections;
using System.Collections.Generic;
using UnityEngine.Rendering;
using UnityEngine;

[RequireComponent (typeof(Light))]
public class CopyLightPass : MonoBehaviour {
    
        CommandBuffer cb = null;
        public RenderTexture m_ShadowmapCopy;


    void OnEnable()
        {

        cb = new CommandBuffer
        {
            name = "CopyLightPass"
        };

        RenderTargetIdentifier shadowmap = BuiltinRenderTextureType.CurrentActive;
        m_ShadowmapCopy = new RenderTexture(1024, 1024, 16, RenderTextureFormat.ARGB32);
        m_ShadowmapCopy.filterMode = FilterMode.Point;

        // Change shadow sampling mode for m_Light's shadowmap.
        cb.SetShadowSamplingMode(shadowmap, ShadowSamplingMode.RawDepth);

        // The shadowmap values can now be sampled normally - copy it to a different render texture.
        var id = new RenderTargetIdentifier(m_ShadowmapCopy);
        cb.Blit(shadowmap, id);

      
        cb.SetGlobalTexture("m_ShadowmapCopy", id);

        Light m_Light = this.GetComponent<Light>();
        // Execute after the shadowmap has been filled.
        m_Light.AddCommandBuffer(LightEvent.AfterShadowMap, cb);

        
      
           
        }
    
    
}
