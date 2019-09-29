// Upgrade NOTE: replaced 'unity_World2Shadow' with 'unity_WorldToShadow'

// Upgrade NOTE: replaced '_LightMatrix0' with 'unity_WorldToLight'

// Upgrade NOTE: replaced '_LightMatrix0' with 'unity_WorldToLight'

// ===================================================================================================================
// Raymarching shader. The shader uses code from two sources. One articles by iq  https://www.iquilezles.org/www/articles/terrainmarching/terrainmarching.htm
// Another source is for the PBR lighting, the lighting is abit of an over kills, https://github.com/Nadrin/PBR/blob/master/data/shaders/hlsl/pbr.hlsl
// ===================================================================================================================


Shader "Unlit/Raymarcher"
{
	SubShader
	{
		// ===================================================================================================================
		// --------------------------------------------------------------
		// Main Pass. 
		Pass
		{
			Tags{ "RenderType" = "Opaque"
			"Queue" = "Geometry +10" 
			"LightMode" = "ForwardBase" }			    // The Light mode tag signals unity to bind the globals we need for lighting
			LOD 100

			Cull Back
			
			CGPROGRAM
			#pragma		vertex vert
			#pragma		fragment frag
			#include	"UnityCG.cginc"
			#include	"Lighting.cginc"	
			#include	"AutoLight.cginc"
			//--------------------------------------------
				// Definitions
			
			struct v2f
			{
				float4 worldPos  : TEXCOORD1;			// used to shoot the rays through the pixels
				float4 vertex    : SV_POSITION;
				float4 screenPos : TEXCOORD2;			// Used to sample the back face pass
			};

			struct ray {
				float3 origin;
				float3 direction;
				float  minmumT;							// the starting point of the ray march
				float  maximumT;						// end point of the ray march
			};
			struct rOut {								// this struct is ommited by reference from the ray marche function
				float  t;								// how far in the march did it hit the surface
				float3 p;								// the point of contact with the surace
			};

			//--------------------------------------------
			// Declerations ------------------------------

			// samplers
			sampler2D_float		_BackFaceRender;			// texture from the command buffer pass rendering the back faces 
			sampler2D			_BlueNoise;					// blue noise used to break percision banding
			sampler2D			m_ShadowmapCopy;			// The depth render of the directional light camera. Is coppied over by a command buffer
			sampler2D			_DebugTexture;				// Used for debuging sufraces. It is agrid with numbers

			float3 ColorOne; 
			float3 ColorTwo;								// the two colors used to control the albedo

			// shading constants
			static const float	PI = 3.141592;
			static const float	Epsilon = 0.00001;
			static const float3 Fdielectric = 0.04;
			float3				_LightDrection;				// The _WorldSpaceLightPos0 is sometimes not set currectly so I am passing this from script
			float3				_LightDrectionRight;		// The orthogonal light coordinate systems used to multisample casted shadows from direction light
			float3				_LightDrectionUp;

			// material properties
			float				roughness = 0.;
			float				metalness = 0.;
	
			// wave constants
			static const float2 _WaveOrigin = float2(0., 0.);		// value used as the origin of the waves. 
			static const float  meanFrequency = .6;
			static const float  baseSpeed = 7.;
			static const float  averageAmplitude = 0.2;
			static const float  numberOfWaves = 16.;
			
			// ===================================================================================================================
			// HELPER FUNCTIONS ----------------------------------------------


			// shading helper functions
		
			float ndfGGX(float cosLh, float roughness)							// normal distribution
			{
				float  alpha   = roughness * roughness;
				float  alphaSq = alpha * alpha;
				float  denom   = (cosLh * cosLh) * (alphaSq - 1.0) + 1.0;
				return alphaSq / (PI * denom * denom);
			}

		
			float gaSchlickG1(float cosTheta, float k)							// Single term for separable Schlick-GGX below.
			{
				return cosTheta / (cosTheta * (1.0 - k) + k);
			}

			
			float gaSchlickGGX(float cosLi, float cosLo, float roughness)		// Schlick-GGX approximation of geometric attenuation function using Smith's method.
			{
				float r = roughness + 1.0;
				float k = (r * r) / 8.0;										// Epic suggests using this roughness remapping for analytic lights.
				return gaSchlickG1(cosLi, k) * gaSchlickG1(cosLo, k);
			}

			
			float3 fresnelSchlick(float3 F0, float cosTheta)					// Shlick's approximation of the Fresnel factor.
			{
				return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
			}


			// Gerstner wave helper Functions -------------------------------------

			float rand(float seed) { return  frac(sin(seed *51.52 + 2.21) *96.2); }

			float SurfaceHeight(float2 xz) {										// this can be replaced by any other height function. such as complex noise based terrain and such

				float2 direction = normalize(xz - _WaveOrigin);						// very neat trick, dot(pos, direction) to get ceneterlized waves
				float y = 0.;
		

				for (float i = 0; i < numberOfWaves; i++) {							// iterate through every wave, use the averages to choose a random wave attribute for each
					float frequency = (rand(i) + 0.5) * meanFrequency;
					float speed = (rand(i + 61.21) + 0.5) * baseSpeed;
					float amplitude = (rand(i + 12.8) + 0.5)* averageAmplitude;

					y += amplitude * sin(frequency* dot(direction.xy, xz.xy)
						+ speed * frequency *-_Time.y);
				}
				return y;
			}

			// ray casting ----------------------------------------------------------

			bool castRay(const ray InRay, out rOut outS, float dta)					// dt value holding the progression through the march
			{													
				      float dt   =  dta;
				const float mint =  InRay.minmumT;									// the starting point of the ray is the pixel depth of the meshes front face
				const float maxt =  InRay.maximumT;									// the end point of the ray is the pixel depth of the meshes back face. 
				      float lh   =  0.0f;											// these two values are used to interpolate between the dt where the function
				      float ly   =  0.0f;											// hits and the previeus march step. It is a filter that attempts to reconstruct a more correct hit point
				        int i    =  0;

				for (float t = mint; t < maxt; t += dt)								// march while within the min max range
				{
					// -------------------------									// safty to make sure no endless loop. I had unity editor freezing on me on certain angles
					i++;															// without this, not quite sure why, but better safe than sorry
					if (i > 100) break;												// this will become an issue if the raymarch volumes are too big
					// -------------------------

					const float3  p = InRay.origin + normalize(InRay.direction) * t;
					const float h = SurfaceHeight(p.xz);

					if (p.y < h)
					{
						
						outS.t = t - dt + dt * (lh - ly) / (p.y - ly - h + lh);		// interpolate the intersection distance in hope of recunstructing a more accurte hitpoint
						outS.p = p;
						return true;
					}

					
					dt = dta *t;													// allow the error to be proportinal to the distance. Further away from the camera we can tolerate higher error
					lh = h;
					ly = p.y;
				}
				return false;
			}

			float3 getNormal(const float3 p)
			{
				float3 n   = float3(0., 0., 0.);
				float  eps = 0.001;
				n.x = SurfaceHeight(float2(p.x - eps, p.z))  - SurfaceHeight(float2(p.x + eps, p.z));
				n.y = 2.0f*eps;
				n.z = SurfaceHeight(float2(p.x, p.z - eps)) - SurfaceHeight(float2(p.x, p.z + eps));

				return normalize(n);
			}


			ray GenerateRay(const float3 cameraPos, const float3 worldPos, float offset, float Back01) {
				ray r;
				r.origin    = cameraPos;
				r.direction = worldPos - cameraPos;
				r.minmumT   = length(r.direction)+ offset;
				r.maximumT  = Back01;
				return r;
			}

			// used to generate the ray for self shadowing in raymarched surfaces. This sends a ray back from it point in the direction
			// of the light
			ray GenerateShadowRay(const float3 lightDir, const float3 start, const float3 surfaceNormal, const float offset, const float end) {
				ray r;
				r.direction = lightDir;
				r.origin    = start + lightDir * offset;
				r.minmumT   = 0.0;
				r.maximumT  = end;
				return r;
			}

			// ===================================================================================================================
			// VERTEX FRAGMENT SHADER

			v2f vert(float4 vertex : POSITION)
			{
				v2f o;
				o.vertex      = UnityObjectToClipPos(vertex);
				o.worldPos    = mul(unity_ObjectToWorld, vertex);
				o.screenPos   = ComputeScreenPos(o.vertex);
				o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);			// old calculation, as I used the depth buffer comparision for min max ray march. I will leave this for future use
				return o;
			}

			void frag (v2f i, out half4 color:SV_Target, out float depth : SV_Depth) 
			{
				fixed4 col = float4(0.,0.,0.,0.);

				// look ups and setups
				// ---------------------------------------------------------------						
				float2 screenUV     = i.screenPos.xy / i.screenPos.w;
				float backFaceDepth = tex2D(_BackFaceRender, screenUV);								// reading the depth value of back face. This will be the end point of our marche				
				float blueNoise     = tex2D(_BlueNoise, screenUV*12.);
				blueNoise           = frac(blueNoise + 0.61803398875 * float(_Time.y % 16));
				// ---------------------------------------------------------------
			
				

				ray r = GenerateRay(_WorldSpaceCameraPos, i.worldPos, blueNoise*0.4, backFaceDepth);
				rOut outS;
			
				// Raymarch
				if (castRay(r, outS, 0.01))															// has hit the surface, shade
				{
					//Setting up base albedo color
					// ---------------------------------------------------------
					float depth = (outS.t - r.minmumT)/(r.maximumT - r.minmumT);	// how far the march went in unit value between min and max 
					col.xyz     = lerp(ColorOne, ColorTwo, saturate(depth));		// doing this to fake volumetric look for now
					
					float3 normal = getNormal(outS.p);
					col.xyz += normal/4.;											// add normal to albedo to make it abit easier to visulaize the topology
					// ----------------------------------------------------------


					// Prepare Shading
					//-----------------------------------------------------------
					float3 Lo    = normalize(r.direction);							// view direction
					float  cosLo = max(0.0, dot(normal, Lo));
					float3 Lr    = 2.0 * cosLo * normal - Lo;
					
					float3 Li = -_LightDrection;									// Light direction. _WorldSpaceLightPos0 is sometimes not set currectly so I am using a costume one
					float3 Lh = normalize(Li + Lo);									// Half-vector between Lo. useful optimization for several lights

					float cosLi = max(0.0, dot(normal, Li));						// Calculate angles between surface normal and various light vectors.
					float cosLh = max(0.0, dot(normal, Lh));
					//-----------------------------------------------------------


					// Shading
					//-----------------------------------------------------------

					float3 F  = fresnelSchlick(Fdielectric, max(0.0, dot(Lh, Lo))); // Calculate Fresnel term for direct lighting.
					float  D  = ndfGGX(cosLh, roughness);						    // Calculate normal distribution for specular BRDF.
					float  G  = gaSchlickGGX(cosLi, cosLo, roughness);			    // Calculate geometric attenuation for specular BRDF.
					float3 kd = lerp(float3(1, 1, 1) - F, float3(0, 0, 0), metalness);

					float3 diffuseBRDF  = kd * col.xyz*depth;						// Cook-Torrance specular microfacet BRDF.
					float3 specularBRDF = (F * D * G) / 
						max(Epsilon, 4.0 * cosLi * cosLo);

					col.xyz += (diffuseBRDF + specularBRDF) * 2. * cosLi;			// add lighting. Hard coded light intesity, light is white
					//-----------------------------------------------------------

						
					// Shadow Casting. Still not working as well as it should. It either needs more noise, better tapping, or more samples
					//-----------------------------------------------------------
					ray rShadow = GenerateShadowRay(Li, outS.p, normal,					// Generate a ray back from the hit position in the direction of the light
						 0.8, 10.);														// p + i is the ray startig point, Li has a length one so displace it back with -1 along Li to start at p (lazy I know, didnt want to change the method)
					rOut  outShadow;
					float tapDevation = 0.9	;
					float dt		  = 0.1;
					float s			  = (float)castRay(rShadow, outShadow, dt);			// if a hit (true -> s=1) it is in shadow
					
					// Displace the direction of the light abit and take 5 samples. 
					// This is an attempt to imitate a more area light like behaviour
					// from the point light. It is not working as it should though.

					rShadow.direction = lerp(_LightDrectionRight, Li, tapDevation);
					GenerateShadowRay(Li, outS.p, rShadow.direction, 0.7, 10.);
					s+= (float)castRay(rShadow, outShadow, dt);
					
					rShadow.direction = lerp(-_LightDrectionRight, Li, tapDevation);
					GenerateShadowRay(Li, outS.p, rShadow.direction, 0.9, 10.);
					s += (float)castRay(rShadow, outShadow, dt);

					rShadow.direction = lerp(_LightDrectionUp, Li, tapDevation);
					GenerateShadowRay(Li, outS.p, rShadow.direction, 0.75, 10.);
					s += (float)castRay(rShadow, outShadow, dt);

					rShadow.direction = lerp(-_LightDrectionUp, Li, tapDevation);
					GenerateShadowRay(Li, outS.p, rShadow.direction, 0.6, 10.);
					s += (float)castRay(rShadow, outShadow, dt);

					col.xyz *= max(0.5 + F,1.-(s/5.));									// I have no ambient lighting term, so I am maxing with 0.4 to avoid a Film Noir look 
					//----------------------------------------------------------- 
				

				}
				else																	// didn't hit the surface, draw the sky box. In unity's pipeline this means just discard the fragment									
				{
					discard;
				}



				// ----------------------------------------------------------------
				// Updating Z-Buffer. Keep in mind this is not supported in all APIs (OpenGL ES), not quite sure what Unity compiles for 
				// those platform. Also it is a slow processs to write back to depth buffer. Also below dx11 there are difference in Semantics. 
				float4 pInScreenSpace = mul(UNITY_MATRIX_VP, float4(outS.p.xyz, 1.));
				depth = pInScreenSpace.z / pInScreenSpace.w;
				// ----------------------------------------------------------------




				// Shadow Coordinates setup for directional light
				// ----------------------------------------------------------------

				//float  vDepth  = distance(_WorldSpaceCameraPos, float4(outS.p.xyz, 1.));			//An alternative to calculate the depth
				float  vDepth = outS.t;																// the distance to camera is used to deterimne which cascaded to use
				
				// The _LightSplitsNear contains the near planes (starts) of each cascaded regions. The four floats correspond to the 
				// begining values of the four cascaded in world space units. The _LightSplitsFar is the same but for the far plane. 
				// so _LightSplitsNear.x and _LightSplitsFar.x give you the first cascaded region from beging to the end in world space units.

				float4 near    = float4 (vDepth >= _LightSplitsNear);								// Checking if the pixel is further away than the near plane for each cascaded				
				float4 far     = float4 (vDepth < _LightSplitsFar);									// same but for closer than far plane for each cascaded
				float4 weights = near * far;														// only have one in xyzw for a depth that is both further away than the near plane and closer than far plane (in subfrusta)

				// Calculate the shadowmap UV coordinate for each cascaded. unity_WorldToShadow is a array of four matrices, 
				// containting the world to camera space transformation matrix for the directional light camera. Each entery of
				// the array is for one of the cascaded cordinate systems. 

				float3 shadowCoord0 = mul(unity_WorldToShadow[0], float4(outS.p.xyz, 1.)).xyz;		
				float3 shadowCoord1 = mul(unity_WorldToShadow[1], float4(outS.p.xyz, 1.)).xyz;
				float3 shadowCoord2 = mul(unity_WorldToShadow[2], float4(outS.p.xyz, 1.)).xyz;
				float3 shadowCoord3 = mul(unity_WorldToShadow[3], float4(outS.p.xyz, 1.)).xyz;


				float3 coord =									// A smart way to avoid branching. Calculating the final shadow texture uv coordinates per fragment
					shadowCoord0 * weights.x +					// case: Cascaded one
					shadowCoord1 * weights.y +					// case: Cascaded two
					shadowCoord2 * weights.z +					// case: Cascaded three
					shadowCoord3 * weights.w;					// case: Cascaded four

				float shadowmask =  tex2D(m_ShadowmapCopy, coord.xy).r;	// We don't need to turn it in linear space since Coord.z is also in log based zbuffer
				
				// standard shadow mapping. If the depth of the pixel in directional light's space (coord.z) is bigger than 
				// the depth of the pixel which the directional light camera is seeing (shadowmask.x), or expressed in other ways,
				// if the pixel's distance to directional light is further away than the closest object the light is seeing,
				// the pixel is behind this occluder and is in shadow. 
				
				// Here a bias is typicly is added (source of peter paning artifacts)
				// to avoid selfshadowing and shadow Acnes. However since the raymarched fragments dont shadow cast through shadowmapping
				// we can ommit that from the calculations.

				// You can use PCF here for anti aliasing. Just make sure to use Unitys macro for it
				// for multi platform and api support, or maybe even getting hardware PCF
				shadowmask = shadowmask > coord.z;			
				col		  *= max(0.3, 1.0 - shadowmask);
				// ----------------------------------------------------------------

				color = col;
				
				return;
			}
			ENDCG
		}
	}
}
