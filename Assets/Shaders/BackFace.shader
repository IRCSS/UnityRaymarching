Shader "Unlit/BackFace"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100
		ZWrite On
		ZTest Always
		// --------------------------------------------------------------------------------------------------------------------------------
		// DEPTH PASS. This renderes the backside of the mesh only. Also it only renders to the depth buffer.

		Pass
	{
		Tags{ "RenderType" = "Opaque" }
		Zwrite On
		
		Cull Front

		CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma multi_compile_shadowcaster
#include "UnityCG.cginc"

		struct v2f {
		float4 vertex : POSITION;
		float4 worldPos : TEXCOORD1;
	};


	v2f  vert(appdata_base v) {
		v2f o;
		o.vertex = UnityObjectToClipPos(v.vertex);
		o.worldPos = mul(unity_ObjectToWorld, v.vertex);
		return o;
	}

	float4 frag(v2f i) : SV_Target{ 
		return distance(_WorldSpaceCameraPos,i.worldPos);
	}

		ENDCG
	}

	}
}
