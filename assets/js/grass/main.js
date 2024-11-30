import * as THREE from 'three';
import { ImprovedNoise } from 'three/addons/math/ImprovedNoise.js';
import { MeshSurfaceSampler } from 'three/addons/math/MeshSurfaceSampler.js';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';

const fastfloor = (x) => {
  return ~~x;
};

const clamp = (x, l, u) => Math.max(l, Math.min(u, x));

const fbm = (noise_fn, octaves, lac, gain) => {
  let amp = 1.0;
  let freq = 1.0;
  let res = 0.0;
  for (let i = 0; i < octaves; ++i) {
    res += amp * noise_fn(freq);
    freq *= lac;
    amp *= gain;
  }
  return res;
}

const frustumContainsPointWithMargin = (frustum, point, lr_margin, ub_margin) => {
  const planes = frustum.planes;
  for (let i = 0; i < 2; i++) {
    if (planes[i].distanceToPoint(point) < -lr_margin) {
      return false;
    }
  }
  for (let i = 2; i < 4; i++) {
    if (planes[i].distanceToPoint(point) < -ub_margin) {
      return false;
    }
  }
  return true;
};

const getNoiseFragmentShader = () => {
  return `
    // https://www.shadertoy.com/view/3dVXDc
    // Hash by David_Hoskins
    #define UI0 1597334673U
    #define UI1 3812015801U
    #define UI2 uvec2(UI0, UI1)
    #define UI3 uvec3(UI0, UI1, 2798796415U)
    #define UIF (1.0 / float(0xffffffffU))

    uniform float uZCoord;

    vec3 hash33(vec3 p) {
      uvec3 q = uvec3(ivec3(p)) * UI3;
      q = (q.x ^ q.y ^ q.z)*UI3;
      return -1. + 2. * vec3(q) * UIF;
    }

    float remap(float x, float a, float b, float c, float d) {
      return (((x - a) / (b - a)) * (d - c)) + c;
    }

    // Gradient noise by iq (modified to be tileable)
    float gradientNoise(vec3 x, float freq) {
      // grid
      vec3 p = floor(x);
      vec3 w = fract(x);
      
      // quintic interpolant
      vec3 u = w * w * w * (w * (w * 6. - 15.) + 10.);

      
      // gradients
      vec3 ga = hash33(mod(p + vec3(0., 0., 0.), freq));
      vec3 gb = hash33(mod(p + vec3(1., 0., 0.), freq));
      vec3 gc = hash33(mod(p + vec3(0., 1., 0.), freq));
      vec3 gd = hash33(mod(p + vec3(1., 1., 0.), freq));
      vec3 ge = hash33(mod(p + vec3(0., 0., 1.), freq));
      vec3 gf = hash33(mod(p + vec3(1., 0., 1.), freq));
      vec3 gg = hash33(mod(p + vec3(0., 1., 1.), freq));
      vec3 gh = hash33(mod(p + vec3(1., 1., 1.), freq));
      
      // projections
      float va = dot(ga, w - vec3(0., 0., 0.));
      float vb = dot(gb, w - vec3(1., 0., 0.));
      float vc = dot(gc, w - vec3(0., 1., 0.));
      float vd = dot(gd, w - vec3(1., 1., 0.));
      float ve = dot(ge, w - vec3(0., 0., 1.));
      float vf = dot(gf, w - vec3(1., 0., 1.));
      float vg = dot(gg, w - vec3(0., 1., 1.));
      float vh = dot(gh, w - vec3(1., 1., 1.));
    
      // interpolation
      return va + 
              u.x * (vb - va) + 
              u.y * (vc - va) + 
              u.z * (ve - va) + 
              u.x * u.y * (va - vb - vc + vd) + 
              u.y * u.z * (va - vc - ve + vg) + 
              u.z * u.x * (va - vb - ve + vf) + 
              u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
    }

    // Tileable 3D worley noise
    float worleyNoise(vec3 uv, float freq) {    
      vec3 id = floor(uv);
      vec3 p = fract(uv);
      
      float minDist = 10000.;
      for (float x = -1.; x <= 1.; ++x) {
        for(float y = -1.; y <= 1.; ++y) {
          for(float z = -1.; z <= 1.; ++z) {
            vec3 offset = vec3(x, y, z);
            vec3 h = hash33(mod(id + offset, vec3(freq))) * .5 + .5;
            h += offset;
            vec3 d = p - h;
            minDist = min(minDist, dot(d, d));
          }
        }
      }
      
      // inverted worley noise
      return 1. - minDist;
    }

    // Fbm for Perlin noise based on iq's blog
    float perlinfbm(vec3 p, float freq, int octaves) {
      float G = exp2(-.85);
      float amp = 1.;
      float noise = 0.;
      for (int i = 0; i < octaves; ++i) {
        noise += amp * gradientNoise(p * freq, freq);
        freq *= 2.;
        amp *= G;
      }
      return noise;
    }

    // Tileable Worley fbm inspired by Andrew Schneider's Real-Time Volumetric Cloudscapes
    // chapter in GPU Pro 7.
    float worleyFbm(vec3 p, float freq) {
      return worleyNoise(p*freq, freq) * .625 +
              worleyNoise(p*freq*2., freq*2.) * .25 +
              worleyNoise(p*freq*4., freq*4.) * .125;
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / resolution.xy;
      vec3 uvw = vec3(uv.xy, uZCoord);
      float freq = 4.;

      float pfbm = mix(1., perlinfbm(uvw, 4., 7), .5);
      pfbm = abs(pfbm * 2. - 1.); // billowy perlin noise

      float w1 = worleyFbm(uvw, freq);
      float w2 = worleyFbm(uvw, freq*2.);
      float w3 = worleyFbm(uvw, freq*4.);
      float w4 = worleyFbm(uvw, freq*8.);
      float perlinWorley = remap(pfbm, 0., 1., w1, 1.);
      float wfbm = w1 * .625 + w2 * .125 + w3 * .25 + w4 * 0.125;

      // cloud shape modeled after the GPU Pro 7 chapter
      float cloud = remap(perlinWorley, wfbm - 1., 1., 0., 1.);
      cloud = remap(cloud, .85, 1., 0., 1.); // fake cloud coverage

      gl_FragColor = vec4(cloud);
    }
  `;
}

const getKuwaharaPass = (radius, display_width, display_height) => { 
  const kuwaharaPass = new ShaderPass(
    {
      name: 'Kuwahara',
      uniforms: {
        'tDiffuse': { value: null },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: `
        precision highp sampler2D;

        const int radius = ${radius};
        uniform float opacity;
        uniform sampler2D tDiffuse;
        varying vec2 vUv;
        vec4 kuwahara(vec2 uv) {
          vec4 fragColor = gl_FragColor;
          vec2 src_size = vec2(1.0 / ${display_width}.0, 1.0 / ${display_height}.0);
          float n = float((radius + 1) * (radius + 1));
          int i;
          int j;
          vec3 m0 = vec3(0.0); vec3 m1 = vec3(0.0); vec3 m2 = vec3(0.0); vec3 m3 = vec3(0.0);
          vec3 s0 = vec3(0.0); vec3 s1 = vec3(0.0); vec3 s2 = vec3(0.0); vec3 s3 = vec3(0.0);
          vec3 c;

          for (int j = -radius; j <= 0; ++j)  {
              for (int i = -radius; i <= 0; ++i)  {
                  c = texture2D(tDiffuse, uv + vec2(i,j) * src_size).rgb;
                  m0 += c;
                  s0 += c * c;
              }
          }

          for (int j = -radius; j <= 0; ++j)  {
              for (int i = 0; i <= radius; ++i)  {
                  c = texture2D(tDiffuse, uv + vec2(i,j) * src_size).rgb;
                  m1 += c;
                  s1 += c * c;
              }
          }

          for (int j = 0; j <= radius; ++j)  {
              for (int i = 0; i <= radius; ++i)  {
                  c = texture2D(tDiffuse, uv + vec2(i,j) * src_size).rgb;
                  m2 += c;
                  s2 += c * c;
              }
          }

          for (int j = 0; j <= radius; ++j)  {
              for (int i = -radius; i <= 0; ++i)  {
                  c = texture2D(tDiffuse, uv + vec2(i,j) * src_size).rgb;
                  m3 += c;
                  s3 += c * c;
              }
          }


          float min_sigma2 = 1e+2;
          m0 /= n;
          s0 = abs(s0 / n - m0 * m0);

          float sigma2 = s0.r + s0.g + s0.b;
          if (sigma2 < min_sigma2) {
              min_sigma2 = sigma2;
              fragColor = vec4(m0, 1.0);
          }

          m1 /= n;
          s1 = abs(s1 / n - m1 * m1);

          sigma2 = s1.r + s1.g + s1.b;
          if (sigma2 < min_sigma2) {
              min_sigma2 = sigma2;
              fragColor = vec4(m1, 1.0);
          }

          m2 /= n;
          s2 = abs(s2 / n - m2 * m2);

          sigma2 = s2.r + s2.g + s2.b;
          if (sigma2 < min_sigma2) {
              min_sigma2 = sigma2;
              fragColor = vec4(m2, 1.0);
          }

          m3 /= n;
          s3 = abs(s3 / n - m3 * m3);

          sigma2 = s3.r + s3.g + s3.b;
          if (sigma2 < min_sigma2) {
              min_sigma2 = sigma2;
              fragColor = vec4(m3, 1.0);
          }
          return fragColor;
        }

        vec4 linearToSRGB(vec4 value) {
          return vec4(mix(pow(value.rgb, vec3(0.55666)) * 1.055 - vec3(0.055), value.rgb * 1.92, vec3(lessThanEqual(value.rgb, vec3(0.0031308)))), value.a);
        }
        void main() {
          gl_FragColor = linearToSRGB(kuwahara(vUv));
        }`
    }
  );
  return kuwaharaPass;
}

const makeDistanceBasedMeshSurfaceSampler = (mesh) => {
  /**
   * Modified MeshSurfaceSampler that samples more points the closer you get to the central point.
   *
   * Building the sampler is a one-time O(n) operation. Once built, any number of
   * random samples may be selected in O(logn) time. Memory usage is O(n).
   *
   * References:
   * - http://www.joesfer.com/?p=84
   * - https://stackoverflow.com/a/4322940/1314762
   */
  const _face = new THREE.Triangle();
  class DistanceBasedMeshSurfaceSampler extends MeshSurfaceSampler {
    constructor(mesh) {
      super(mesh);
      this.distanceAttribute = null;
    }
    // Distance-based weighting
    setDistanceAttribute(centralPoint) {
      this.distanceAttribute = centralPoint;
      return this;
    }
    getSquaredDistance(positionBuffer, i0, i1, i2) {
      const xDiff = (positionBuffer.getX(i0) + positionBuffer.getX(i1) + positionBuffer.getX(i2)) / 3 - this.distanceAttribute.x;
      const zDiff = (positionBuffer.getZ(i0) + positionBuffer.getZ(i1) + positionBuffer.getZ(i2)) / 3  - this.distanceAttribute.z;
      return 1 / (Math.pow(xDiff * xDiff + zDiff * zDiff, 0.48) + 1.0);
    }
    build() {
      const indexAttribute = this.indexAttribute;
      const positionAttribute = this.positionAttribute;
      const weightAttribute = this.weightAttribute;
      const distanceAttribute = this.distanceAttribute;

      const totalFaces = indexAttribute ? (indexAttribute.count / 3) : (positionAttribute.count / 3);
      const faceWeights = new Float32Array(totalFaces);

      // Accumulate weights for each mesh face.
      for (let i = 0; i < totalFaces; i ++ ) {
        let faceWeight = 1;

        let i0 = 3 * i;
        let i1 = 3 * i + 1;
        let i2 = 3 * i + 2;

        if (indexAttribute) {
          i0 = indexAttribute.getX(i0);
          i1 = indexAttribute.getX(i1);
          i2 = indexAttribute.getX(i2);
        }

        if ( weightAttribute ) {
          faceWeight = weightAttribute.getX( i0 )
            + weightAttribute.getX( i1 )
            + weightAttribute.getX( i2 );
        } else if ( distanceAttribute ) {
          faceWeight = this.getSquaredDistance(this.geometry.getAttribute('position'), i0, i1, i2);
        }

        _face.a.fromBufferAttribute( positionAttribute, i0 );
        _face.b.fromBufferAttribute( positionAttribute, i1 );
        _face.c.fromBufferAttribute( positionAttribute, i2 );
        faceWeight *= _face.getArea();

        faceWeights[ i ] = faceWeight;
      }

      // Store cumulative total face weights in an array, where weight index
      // corresponds to face index.

      const distribution = new Float32Array(totalFaces);
      let cumulativeTotal = 0;
      for ( let i = 0; i < totalFaces; i ++ ) {
        cumulativeTotal += faceWeights[ i ];
        distribution[ i ] = cumulativeTotal;
      }
      this.distribution = distribution;
      return this;
    }
  }
  return new DistanceBasedMeshSurfaceSampler(mesh);
};

const makeGPUComputationRenderer3D = (sizeX, sizeY, sizeZ, renderer) => {
  /** Modified GPUComputationRenderer class that writes to a 3D texture instead of a 2D one */
  class GPUComputationRenderer3D {
    constructor(sizeX, sizeY, sizeZ, renderer) {
      this.variables = [];
      this.currentTextureIndex = 0;
      let dataType = THREE.FloatType;
      const passThruUniforms = {
        passThruTexture: { value: null }
      };
      const passThruShader = createShaderMaterial(getPassThroughFragmentShader(), passThruUniforms);
      const quad = new FullScreenQuad(passThruShader);
      this.setDataType = function (type) {
        dataType = type;
        return this;
      };
      this.addVariable = function (variableName, computeFragmentShader, initialValueTexture) {
        const material = createShaderMaterial(computeFragmentShader, {uZCoord: {value: 0.0}});
        const variable = {
          name: variableName,
          initialValueTexture: initialValueTexture,
          material: material,
          dependencies: null,
          renderTargets: [],
          wrapS: THREE.RepeatWrapping,
          wrapT: THREE.RepeatWrapping,
          wrapR: THREE.RepeatWrapping,
          minFilter: THREE.LinearFilter,
          magFilter: THREE.LinearFilter
        };
        this.variables.push(variable);
        return variable;
      };

      this.setVariableDependencies = function (variable, dependencies) {
        variable.dependencies = dependencies;
      };

      this.init = function () {
        if (renderer.capabilities.maxVertexTextures === 0) {
          return 'No support for vertex shader textures.';
        }
        for (let i = 0; i < this.variables.length; ++i) {
          const variable = this.variables[i];
          // Creates rendertargets and initialize them with input texture
          variable.renderTargets[0] = this.createRenderTarget(sizeX, sizeY, sizeZ, variable.wrapS, variable.wrapT, variable.wrapR, variable.minFilter, variable.magFilter);
          variable.renderTargets[1] = this.createRenderTarget(sizeX, sizeY, sizeZ, variable.wrapS, variable.wrapT, variable.wrapR, variable.minFilter, variable.magFilter);
          this.renderTexture(variable.initialValueTexture, variable.renderTargets[0]);
          this.renderTexture(variable.initialValueTexture, variable.renderTargets[1]);

          // Adds dependencies uniforms to the ShaderMaterial
          const material = variable.material;
          const uniforms = material.uniforms;

          if ( variable.dependencies !== null ) {
            for ( let d = 0; d < variable.dependencies.length; d ++ ) {
              const depVar = variable.dependencies[ d ];
              if ( depVar.name !== variable.name ) {
                // Checks if variable exists
                let found = false;
                for ( let j = 0; j < this.variables.length; j ++ ) {
                  if ( depVar.name === this.variables[ j ].name ) {
                    found = true;
                    break;
                  }
                }
                if (!found) {
                  return 'Variable dependency not found. Variable=' + variable.name + ', dependency=' + depVar.name;
                }
              }
              uniforms[ depVar.name ] = { value: null };
              material.fragmentShader = '\nuniform sampler3D ' + depVar.name + ';\n' + material.fragmentShader;
            }
          }
        }
        this.currentTextureIndex = 0;
        return null;

      };

      this.compute = function () {
        const currentTextureIndex = this.currentTextureIndex;
        const nextTextureIndex = this.currentTextureIndex === 0 ? 1 : 0;

        for (let i = 0, il = this.variables.length; i < il; ++i) {
          const variable = this.variables[i];
          // Sets texture dependencies uniforms
          if (variable.dependencies !== null) {
            const uniforms = variable.material.uniforms;
            for (let d = 0, dl = variable.dependencies.length; d < dl; ++d) {
              const depVar = variable.dependencies[d];
              uniforms[depVar.name].value = depVar.renderTargets[currentTextureIndex].texture;
            }
          }

          // Performs the computation for this variable
          this.doRenderTarget( variable.material, variable.renderTargets[nextTextureIndex]);
        }
        this.currentTextureIndex = nextTextureIndex;
      };

      this.getCurrentRenderTarget = function ( variable ) {
        return variable.renderTargets[this.currentTextureIndex];
      };

      this.getAlternateRenderTarget = function ( variable ) {
        return variable.renderTargets[this.currentTextureIndex === 0 ? 1 : 0];
      };

      this.dispose = function () {
        quad.dispose();
        const variables = this.variables;
        for (let i = 0; i < variables.length; i++) {
          const variable = variables[ i ];
          if ( variable.initialValueTexture ) variable.initialValueTexture.dispose();
          const renderTargets = variable.renderTargets;
          for ( let j = 0; j < renderTargets.length; j ++ ) {
            const renderTarget = renderTargets[ j ];
            renderTarget.dispose();
          }
        }
      };

      function addResolutionDefine(materialShader) {
        materialShader.defines.resolution = 'vec3( ' + sizeX.toFixed(1) + ', ' + sizeY.toFixed(1) + ', ' + sizeZ.toFixed(1) + ' )';
      }

      this.addResolutionDefine = addResolutionDefine;

      // The following functions can be used to compute things manually

      function createShaderMaterial(computeFragmentShader, uniforms) {
        uniforms = uniforms || {};
        const material = new THREE.ShaderMaterial({
          name: 'GPUComputationShader',
          uniforms: uniforms,
          vertexShader: getPassThroughVertexShader(),
          fragmentShader: computeFragmentShader
        });
        addResolutionDefine(material);
        return material;
      }

      this.createShaderMaterial = createShaderMaterial;

      this.createRenderTarget = function (sizeXTexture, sizeYTexture, sizeZTexture, wrapS, wrapT, wrapR, minFilter, magFilter) {
        sizeXTexture = sizeXTexture || sizeX;
        sizeYTexture = sizeYTexture || sizeY;
        sizeZTexture = sizeZTexture || sizeZ;

        wrapS = wrapS || RepeatWrapping;
        wrapT = wrapT || RepeatWrapping;
        wrapR = wrapR || RepeatWrapping;

        minFilter = minFilter || LinearFilter;
        magFilter = magFilter || LinearFilter;

        const renderTarget = new THREE.WebGL3DRenderTarget(sizeXTexture, sizeYTexture, sizeZTexture, {
          wrapS: wrapS,
          wrapT: wrapT,
          wrapR: wrapR,
          minFilter: minFilter,
          magFilter: magFilter,
          format: THREE.RGBAFormat,
          type: dataType,
          depthBuffer: true
        });
        renderTarget.textures[0].wrapS = wrapS;
        renderTarget.textures[0].wrapT = wrapT;
        renderTarget.textures[0].wrapR = wrapR;
        renderTarget.textures[0].minFilter = minFilter;
        renderTarget.textures[0].magFilter = magFilter;
        return renderTarget;
      };

      this.createTexture = function () {
        const data = new Float32Array(sizeX * sizeY * sizeZ * 4);
        const texture = new THREE.Data3DTexture(data, sizeX, sizeY, sizeZ);
        texture.type = THREE.FloatType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.wrapR = THREE.RepeatWrapping;
        texture.needsUpdate = true;
        return texture;
      };

      this.renderTexture = function (input, output) {
        // Takes a texture, and render out in rendertarget
        // input = Texture
        // output = RenderTarget
        passThruUniforms.passThruTexture.value = input;
        this.doRenderTarget(passThruShader, output);
        passThruUniforms.passThruTexture.value = null;
      };

      this.doRenderTarget = function (material, output) {
        const currentRenderTarget = renderer.getRenderTarget();

        const currentXrEnabled = renderer.xr.enabled;
        const currentShadowAutoUpdate = renderer.shadowMap.autoUpdate;

        renderer.xr.enabled = false; // Avoid camera modification
        renderer.shadowMap.autoUpdate = false; // Avoid re-computing shadows
        if (Object.hasOwn(material.uniforms, 'uZCoord')) {
          for (let i = 0; i < sizeZ; ++i) {
            material.uniforms.uZCoord.value = i / (sizeZ - 1.0);
            quad.material = material;
            renderer.setRenderTarget(output, i);
            quad.render(renderer);
          }
          material.uniforms.uZCoord.value = 0.0;
        } else {
          quad.material = material;
          renderer.setRenderTarget(output);
          quad.render(renderer);
        }
        quad.material = passThruShader;

        renderer.xr.enabled = currentXrEnabled;
        renderer.shadowMap.autoUpdate = currentShadowAutoUpdate;

        renderer.setRenderTarget(currentRenderTarget);
      };

      // Shaders
      function getPassThroughVertexShader() {
        return  'void main() {\n' +
            '\n' +
            ' gl_Position = vec4( position, 1.0 );\n' +
            '\n' +
            '}\n';
      }

      function getPassThroughFragmentShader() {
        return  'uniform sampler3D passThruTexture;\n' +
            '\n' +
            'void main() {\n' +
            '\n' +
            ' vec3 uvw = gl_FragCoord.xyz / resolution.xyz;\n' +
            '\n' +
            ' gl_FragColor = texture(passThruTexture, uvw);\n' +
            '\n' +
            '}\n';
      }
    }
  }
  return new GPUComputationRenderer3D(sizeX, sizeY, sizeZ, renderer);
}

const makeSky = () => {
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 32;

  const context = canvas.getContext('2d');
  const gradient = context.createLinearGradient(0, 0, 0, 32);
  gradient.addColorStop(0.32, '#014a84');
  gradient.addColorStop(0.38, '#0561a0');
  gradient.addColorStop(0.55, '#ace4f8');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 1, 32);
  const skymap = new THREE.CanvasTexture(canvas);
  skymap.colorSpace = THREE.SRGBColorSpace;
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(10000),
    new THREE.MeshBasicMaterial({map: skymap, side: THREE.BackSide, fog: false})
  );
  return sky;
}

const makePlane = () => {
  const planeShader = new THREE.ShaderMaterial({
    vertexShader: `
    precision mediump float;

    varying vec3 vNormal;
    varying vec3 vFragPos;

    void main() {
      vFragPos = vec3(modelMatrix * vec4(position, 1.0));
      vNormal = normalize(normalMatrix * normal);

      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
    fragmentShader: `
    precision mediump float;

    uniform vec3 uLightPosition;  // Light position in world space
    uniform vec3 uLightColor;     // Light color
    uniform vec3 uViewPosition;   // Camera position in world space
    uniform vec3 uAmbientColor;   // Ambient color
    uniform vec3 uDiffuseColor;   // Diffuse color of the material
    uniform vec3 uSpecularColor;  // Specular color of the material
    uniform float uShininess;     // Shininess exponent for the material

    varying vec3 vNormal;         // Normal vector in world space
    varying vec3 vFragPos;        // Fragment position in world space

    void main() {
      vec3 norm = normalize(vNormal);
      vec3 lightDir = normalize(uLightPosition - vFragPos);
      vec3 viewDir = normalize(uViewPosition - vFragPos);

      float distance = length(uViewPosition - vFragPos);

      vec3 ambient = uAmbientColor * uDiffuseColor;

      float diff = max(dot(norm, lightDir), 0.9);
      vec3 diffuse = diff * uLightColor * uDiffuseColor;

      // Specular lighting (Blinn-Phong model)
      vec3 halfwayDir = normalize(lightDir + viewDir);
      float spec = pow(max(dot(norm, halfwayDir), 0.0), uShininess);
      vec3 specular = spec * uLightColor * uSpecularColor;

      vec3 color = ambient + diffuse + specular;

      gl_FragColor = vec4(color * (1.0 - min(200.0 / distance, 0.4)), 1.0);
    }`,
    uniforms: {
      uLightPosition: { value: SUNLIGHT.position },
      uLightColor: { value: SUNLIGHT.color },
      uViewPosition: { value: CAMERA.position },
      uAmbientColor: { value: new THREE.Color(0.2, 0.2, 0.2) },
      uDiffuseColor: { value: new THREE.Color(0.886, 0.827, 0.686) },
      uSpecularColor: { value: new THREE.Color(0.0, 0.0, 0.0) },
      uShininess: { value: 32.0 }
    }
  });

  const geometry = new THREE.PlaneGeometry(10000, 10000, 100, 100);

  const plane = new THREE.Mesh(geometry, planeShader);
  plane.geometry.rotateX(-Math.PI / 2);

  const positionAttribute = plane.geometry.attributes.position;

  // Modify each vertex position
  for (let i = 0; i < positionAttribute.count; i++) {
      // Get the current vertex position
      const x = positionAttribute.getX(i);
      const y = positionAttribute.getY(i);
      const z = positionAttribute.getZ(i);
      // Set the new vertex position (for example, adding an offset)
      positionAttribute.setXYZ(i, x, y + PERLIN.noise(x * 0.0004, z * 0.0008, 1.0) * 165, z);
  }
  positionAttribute.needsUpdate = true;
  plane.geometry.computeVertexNormals();

  return plane;
};

const makeGrass = (
  surface,
  height, 
  width, // controls base width
  cloud_noise_texture,
) => {
  const grassVertShader = `
  #define PI 3.1415926538

  precision highp float;

  const float min_r_inv = 0.0001;
  const float max_r_inv = 0.7;

  uniform float uTime;

  varying vec2 instancePos;
  varying vec3 vNewPos;
  varying vec3 vFragPos;
  varying vec3 vColor;

  //
  // Description : Array and textureless GLSL 2D/3D/4D simplex 
  //               noise functions.
  //      Author : Ian McEwan, Ashima Arts.
  //  Maintainer : stegu
  //     Lastmod : 20201014 (stegu)
  //     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
  //               Distributed under the MIT License. See LICENSE file.
  //               https://github.com/ashima/webgl-noise
  //               https://github.com/stegu/webgl-noise
  // 

  vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
  }

  vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
  }

  vec4 permute(vec4 x) {
       return mod289(((x*34.0)+10.0)*x);
  }

  vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
  }

  float snoise(vec3 v) { 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

  // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

  // Permutations
    i = mod289(i); 
    vec4 p = permute( permute( permute( 
               i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
             + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  // Gradients: 7x7 points over a square, mapped onto an octahedron.
  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

  //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

  // Mix final noise value
    vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 105.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                  dot(p2,x2), dot(p3,x3) ) );
    }

  float perlin(vec2 xy, float seed) { return 0.7 * snoise(vec3(xy, seed)); }
  float perlinTime(vec2 xy) { return 0.7 * snoise(vec3(xy, 4. * uTime)); }

  float scalePerlinTo(float x, float lower, float higher) {
    return (x + 1.0) / 2.0  * (higher - lower) + lower;
  }
  float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
  }

  void main() {
    instancePos = vec2(instanceMatrix[3][0], instanceMatrix[3][2]);

    vec2 step = vec2(1.3, 1.3);
    float r_inv = perlinTime(instancePos * 0.05);
    vec2 wind_param = instancePos * 0.00028 + uTime * 0.36;
    float wind_perlin = perlin(wind_param, 123.0);
    wind_perlin += 0.5 * perlin(wind_param * 3.0 - step, 0.0);
    wind_perlin += 0.25 * perlin(wind_param * 6.0 - step, 0.0);
    wind_perlin = scalePerlinTo(wind_perlin, -0.7, 1.0);
    r_inv = wind_perlin * scalePerlinTo(r_inv, min_r_inv, max_r_inv);
    r_inv += (rand(instancePos) - 0.5) / 10.0;

    float k = position.y / ${height}.0;
    float rad = k * r_inv;

    vColor = instanceColor * (0.38 * tanh(10.0 * k) + 0.62);
    // vColor = (vec3(wind_perlin,wind_perlin,wind_perlin) + 1.0) / 2.0;
    float scaleX = scalePerlinTo(perlin(instancePos, 123.0), 0.04, 0.3);
    float scaleY = scalePerlinTo(perlin(instancePos * 0.08, 1293.0), 0.05, 0.2);
    float x, y;
    if (abs(r_inv) >= min_r_inv) {
      x = 1.0 / r_inv * (1.0 - cos(rad)) * ${height}.0 * 5.0 + position.x;
      y = 1.0 / r_inv * sin(rad) * ${height}.0 + position.y;
    } else {
      float t = (r_inv + min_r_inv) / (2.0 * min_r_inv);
      x = (-1.0 / min_r_inv * (1.0 - cos(k * -min_r_inv)) * ${height}.0 * 5.0) * (1.0 - t) + (1.0 / min_r_inv * (1.0 - cos(k * min_r_inv)) * ${height}.0 * 5.0) * (t);
      x += position.x;
      y = (-1.0 / min_r_inv * sin(k * -min_r_inv) * ${height}.0) * (1.0 - t) + (1.0 / min_r_inv * sin(k * min_r_inv) * ${height}.0) * (t);
      y += position.y;
    }
    vec3 new_position = vec3(x * scaleX, y * scaleY, position.z);

    vNewPos = new_position;

    vFragPos = vec3(modelMatrix * instanceMatrix * vec4(new_position, 1.0));

    gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(new_position, 1.0);
  }`;
  const grassFragShader = `
  precision highp float;

  uniform sampler3D uCloudNoiseTexture;
  uniform float uTime;

  uniform vec3 uLightPosition;  // Light position in world space
  uniform vec3 uLightColor;     // Light color
  uniform vec3 uViewPosition;   // Camera position in world space
  uniform vec3 uAmbientColor;   // Ambient color
  uniform vec3 uDiffuseColor;   // Diffuse color of the material
  uniform vec3 uSpecularColor;  // Specular color of the material
  uniform float uShininess;     // Shininess exponent for the material

  varying vec2 instancePos;
  varying vec3 vNewPos;         // Normal vector in world space
  varying vec3 vFragPos;        // Fragment position in world space
  varying vec3 vColor;

  float sample1(vec2 p) {
    vec3 np = vec3(p.x, 0.0, p.y) - uTime * vec3(0.025, 0.0, 0.02);
    float shadow = texture(uCloudNoiseTexture, np * 1.3).x;
    return shadow;
  }

  void main() {
    // recalc normals from our displacement
    vec3 dx = dFdx(vNewPos);
    vec3 dy = dFdy(vNewPos);
    vec3 norm = normalize(cross(dx, dy));

    vec3 lightDir = normalize(uLightPosition - vFragPos);
    vec3 viewDir = normalize(uViewPosition - vFragPos);
    
    vec3 ambient = uAmbientColor * vColor;

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * uLightColor * vColor;

    // Specular lighting (Blinn-Phong model)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), uShininess);
    vec3 specular = spec * uLightColor * uSpecularColor;

    vec3 color = (1. - min(sample1(instancePos.xy * 0.0002), 0.45)) * ambient + diffuse + specular;

    gl_FragColor = vec4(color, 1.0);
  }`;

  const grassShader = new THREE.ShaderMaterial({
    vertexShader: grassVertShader,
    fragmentShader: grassFragShader,
    uniforms: {
      uCloudNoiseTexture: { value: cloud_noise_texture },
      uLightPosition: { value: SUNLIGHT.position },
      uLightColor: { value: SUNLIGHT.color },
      uViewPosition: { value: CAMERA.position },
      uAmbientColor: { value: new THREE.Color(0.5, 0.5, 0.5) },
      uDiffuseColor: { value: new THREE.Color(1, 0, 0) },
      uSpecularColor: { value: new THREE.Color(0.1, 0.1, 0.1) },
      uShininess: { value: 32.0 },
      uTime: { value: CLOCK.getElapsedTime() }
    }
  });

  const x = 0, y = 0;
  const grassShape = new THREE.Shape();
  // bottom right
  grassShape.moveTo(x + width, y);
  // straight line to bottom left
  grassShape.bezierCurveTo(
    x + width / 3, y,
    x + width / 3, y,
    x, y
  );
  // curve upwards
  grassShape.bezierCurveTo(
    x - width / 3, y + height / 2,
    x, y + height / 2,
    x + width / 2, y + height
  );
  // curve downwards
  grassShape.bezierCurveTo(
    x + width / 1, y + height / 2,
    x + width / 3, y + height / 2,
    x + width, y
  );
  // const geometry = new ExtrudeGeometry(grassShape, {
  //   curveSegments: 6,
  //   depth: 0.1,
  //   bevelEnabled: false
  // });
  const geometry = new THREE.ShapeGeometry(grassShape, 10);
  const grass = new THREE.InstancedMesh(geometry, grassShader, NUM_GRASS);
  grass.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  const grassPaletteDarker = [
    new THREE.Color().setHex(0xa4c656),
    new THREE.Color().setHex(0xa5c642),
    new THREE.Color().setHex(0xabce5d),
    new THREE.Color().setHex(0xd5c733),
  ];
  const grassPaletteLighter = [
    new THREE.Color(0.643, 0.776, 0.337, 1.0),
    new THREE.Color(0.647, 0.776, 0.259, 1.0),
    new THREE.Color(0.671, 0.808, 0.365, 1.0),
    new THREE.Color(0.835, 0.780, 0.200, 1.0),
  ];

  const grassPalette = [];
  for (let i = 0; i < grassPaletteDarker.length; ++i) {
    const _color = new THREE.Color();
    _color.lerpColors(grassPaletteDarker[i], grassPaletteLighter[i], 0.54);
    grassPalette.push(_color);
  }

  const sampler = makeDistanceBasedMeshSurfaceSampler(surface).setDistanceAttribute(CAMERA.position).build();
  const _position = new THREE.Vector3();
  const _normal = new THREE.Vector3();
  const _dummy = new THREE.Object3D();

  const _color = new THREE.Color();
  for (let i = 0; i < NUM_GRASS; i++) {
    sampler.sample(_position, _normal);
    // visibility culling
    while(!frustumContainsPointWithMargin(FRUSTUM, _position, 10, height * 0.7)) {
      sampler.sample(_position, _normal);
    }
    _dummy.position.copy(_position);
    _dummy.lookAt(_normal);
    _dummy.updateMatrix();
    grass.setMatrixAt(i, _dummy.matrix);
    let x = _position.x * 0.005 - 30;
    let y = _position.z * 0.009 + 0.3;
    let z = 80.0;
    let p = fbm((freq) => PERLIN.noise(x * freq, y * freq, z), 4, 2.0, 0.5);
    p = (p + 1.0) / 2.0 * 0.9649214285521897;
    const fractionalIdx = p * (grassPalette.length - 1);
    const leftIdx = clamp(fastfloor(fractionalIdx), 0, grassPalette.length - 1);
    const rightIdx = clamp(leftIdx + 1, 0, grassPalette.length - 1);
    const t = Math.pow(clamp(fractionalIdx % 1, 0, 1), 0.8); // pow here is for smoother transitions
    _color.lerpColors(grassPalette[leftIdx], grassPalette[rightIdx], t);
    grass.setColorAt(i, _color);
  }
  grass.instanceMatrix.needsUpdate = true;
  grass.instanceColor.needsUpdate = true;

  return grass;
}

const makeClouds = (noise_texture) => {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      base: { value: new THREE.Color( 0x798aa0 ) },
      textureNoise: { value: noise_texture },
      threshold: { value: 0.4 },
      opacity: { value: 0.8 },
      range: { value: 0.1 },
      steps: { value: 15 },
      frame: { value: 4.5 },
    },
    vertexShader: `
    varying vec3 vOrigin;
    varying vec3 vDirection;
    varying vec3 vPosition;

    void main() {
      vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
      vPosition = position;
      vDirection = position - vOrigin;

      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
    `,
    fragmentShader: `
    precision highp float;
    precision highp sampler3D;

    varying vec3 vOrigin;
    varying vec3 vDirection;
    varying vec3 vPosition;

    uniform sampler3D textureNoise;
    uniform vec3 base;

    uniform float threshold;
    uniform float range;
    uniform float opacity;
    uniform int steps;
    uniform float frame;

    uint wang_hash(uint seed) {
      seed = (seed ^ 61u) ^ (seed >> 16u);
      seed *= 9u;
      seed = seed ^ (seed >> 4u);
      seed *= 0x27d4eb2du;
      seed = seed ^ (seed >> 15u);
      return seed;
    }

    float randomFloat(uint seed) {
      return float(wang_hash(seed)) / 4294967296.;
    }

    vec2 hitBox(vec3 orig, vec3 dir) {
      const vec3 box_min = vec3(-0.5);
      const vec3 box_max = vec3(0.5);
      vec3 inv_dir = 1.0 / dir;
      vec3 tmin_tmp = (box_min - orig) * inv_dir;
      vec3 tmax_tmp = (box_max - orig) * inv_dir;
      vec3 tmin = min(tmin_tmp, tmax_tmp);
      vec3 tmax = max(tmin_tmp, tmax_tmp);
      float t0 = max(tmin.x, max(tmin.y, tmin.z));
      float t1 = min(tmax.x, min(tmax.y, tmax.z));
      return vec2(t0, t1);
    }

    float sample1(vec3 p) {
      vec3 np = p - frame * vec3(0.00625, 0.00125, 0.025);
      float cloud = texture(textureNoise, np * 1.3).x;
      // height gradient
      float h = vPosition.y + 0.5;
      if (h < 0.1) {
        cloud *= max(pow(1. - pow(abs(10. * h - 1.0), 3.), 3.), 0.); 
      } else {
        cloud *= -pow(2.1 * (h - 0.1), 3.) + 1.;
      }
      return cloud;
    }

    vec4 linearToSRGB(vec4 value) {
      return vec4(mix(pow(value.rgb, vec3(0.41666)) * 1.055 - vec3(0.055), value.rgb * 12.92, vec3(lessThanEqual(value.rgb, vec3(0.0031308)))), value.a);
    }

    float shading(vec3 coord) {
      float step = 0.01;
      return sample1(coord + vec3(-step)) - sample1(coord + vec3(step));
    }

    void main() {
      vec3 rayDir = normalize(vDirection);
      vec2 bounds = hitBox(vOrigin, rayDir);

      if (bounds.x > bounds.y) discard;

      bounds.x = max(bounds.x, 0.0);

      vec3 p = vOrigin + bounds.x * rayDir;
      vec3 inc = 1.0 / abs(rayDir);
      float delta = min(inc.x, min(inc.y, inc.z));
      delta /= float(steps);

      // Jitter
      // Nice little seed from
      // https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
      uint seed = uint(gl_FragCoord.x) * uint(1973) + uint(gl_FragCoord.y) * uint(9277) + uint(frame) * uint(26699);
      vec3 size = vec3(textureSize(textureNoise, 0));
      float randNum = randomFloat(seed) * 2.0 - 1.0;
      p += rayDir * randNum * (0.5 / size);

      vec4 ac = vec4(base, 0.0);
      for (int i=0; i<steps && ac.a<0.96; ++i, p+=rayDir*delta) {
        float d = sample1(p + 0.5);
        d = smoothstep(threshold - range, threshold + range, d) * opacity;
        float col = shading(p + 0.5) * 3.0 + ((p.x + p.y) * 0.25) + 0.2;
        ac.rgb += (1.0 - ac.a) * d * col;
        ac.a += (1.0 - ac.a) * d;
      }
      ac = clamp(ac, 0.0, 1.0);

      if (ac.a == 0.0) discard;

      gl_FragColor = ac;

      float brightness = 0.2;
      float contrast = 0.8;
      gl_FragColor.rgb = clamp(contrast * gl_FragColor.rgb + brightness, 0.0, 1.0);
      float gamma = 0.4;
      gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(gamma));
      gl_FragColor.b *= 1.59;
      gl_FragColor.r *= 0.86;

      gl_FragColor = linearToSRGB(gl_FragColor);
    }
    `,
    side: THREE.BackSide,
    transparent: true
  });

  return new THREE.Mesh(geometry, material);
};

const makeMountains = () => {
  const getMountainHeightMapTexture = () => {
    /** Runs the DLA algorithm */
    const canvas = document.createElement("canvas");
    canvas.width = 10000;
    canvas.height = 10000;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    const getidx = (size) => fastfloor(Math.random() * size);
    const getdir = () => fastfloor(Math.random() * 4);
    const withinbounds = (i, j, size) => i>=0 && j>=0 && i<curr_size && j<curr_size
    const hasneighbours = (i, j, arr) => {
      const size = arr.length;
      let has_neighbour = false;
      for (let k = 0; k < 4; ++k) {
        let ni = i + dirs[k];
        let nj = j + dirs[k + 1];
        has_neighbour |= withinbounds(ni, nj, size) && arr[ni][nj] == 1;
      }
      return has_neighbour;
    };
    const dirs = [0, 1, 0, -1, 0];
    const INIT_SIZE = 5;
    const FILL_FACTOR = 0.1;
    const NUM_EXPANDS = 7;

    let curr_size = INIT_SIZE;
    let arr = Array.from(Array(curr_size), _ => Array(curr_size).fill(0));

    let filled = 0;

    for (let num_expand = 0; num_expand <= NUM_EXPANDS; ++ num_expand) {
      let i = getidx(curr_size);
      let j = getidx(curr_size);
      arr[i][j] = 1;
      while (filled < curr_size * curr_size * FILL_FACTOR) {
        do {
          i = getidx(curr_size);
          j = getidx(curr_size);
        } while (arr[i][j] != 0);
        
        while (!hasneighbours(i, j, arr)) {
          // go around randomly until u hit a neighbour
          let d;
          do {
              d = getdir();
          } while (!withinbounds(i + dirs[d], j + dirs[d + 1]));
          i += dirs[d];
          j += dirs[d + 1];
        }
        arr[i][j] = 1;
        filled += 1;
      }
      // add to canvas while blurring using small gaussian filter
      const imageData = ctx.getImageData(0, 0, curr_size, curr_size);
      const data = imageData.data;
      for (let i = 1; i < curr_size - 1; ++i) {
        for (let j = 1; j < curr_size - 1; ++j) {
          const k = (i * curr_size + j) * 4;
          let c = arr[i][j];
          data[k] = Math.min(255, data[k] + fastfloor(c * 255 / NUM_EXPANDS)); // red
          data[k+1] = Math.min(255, data[k+1] + fastfloor(c * 255 / NUM_EXPANDS)); // green
          data[k+2] = Math.min(255, data[k+2] + fastfloor(c * 255 / NUM_EXPANDS)); // blue
          data[k+3] = 255;
        }
      }
      // ctx.filter = 'blur(1.3px)';
      ctx.putImageData(imageData, 0, 0);
      if (num_expand == NUM_EXPANDS) break;
      // scale up canvas
      ctx.drawImage(canvas, 0, 0, curr_size, curr_size, 0, 0, curr_size * 2 - 1, curr_size * 2 - 1);

      // ARR EXPANSION
      curr_size = curr_size * 2 - 1;
      const new_arr = Array.from(Array(curr_size), _ => Array(curr_size).fill(0));
      filled = 0;
      
      for (let a = 0; a < curr_size; ++a) {
        const og_a = a >> 1;
        const a_in_btw = (a % 2) === 1;
        for (let b = 0; b < curr_size; ++b) {
          const og_b = b >> 1;
          const b_in_btw = (b % 2) === 1;
          if (a_in_btw && b_in_btw) continue;
          if (a_in_btw) {
            if (arr[og_a][og_b] == 1 && arr[og_a + 1][og_b] == 1) {
              const r = fastfloor(Math.random() * 3) - 1;
              new_arr[a][Math.max(Math.min(b + r, curr_size - 1), 0)] = 1;
            }
          }
          else if (b_in_btw) {
            if (arr[og_a][og_b] == 1 && arr[og_a][og_b + 1] == 1) {
              const r = fastfloor(Math.random() * 3) - 1;
              new_arr[Math.max(Math.min(a + r, curr_size - 1), 0)][b] = 1;
            }
          }
          else {
            new_arr[a][b] = arr[og_a][og_b];
          }
          filled += new_arr[a][b];
        }
      }
      arr = new_arr;
    }
    const data = ctx.getImageData(0, 0, curr_size, curr_size).data;
    const texture_data = new Uint8Array(4 * curr_size * curr_size);
    let max_val = 0;
    const max_dist = (curr_size / 2) ** 2 * 1.1;
    for (let i = 0; i < curr_size; ++i) {
      for (let j = 0; j < curr_size; ++j) {
        const stride = (i * curr_size + j) * 4;
        const dist = (i - curr_size / 2.8) ** 2 + (j - curr_size / 2) ** 2;
        const factor = Math.max(1. - (dist / max_dist), 0);
        texture_data[stride] = fastfloor(factor * data[stride]);
        texture_data[stride + 1] = fastfloor(factor * data[stride + 1]);
        texture_data[stride + 2] = fastfloor(factor * data[stride + 2]);
        texture_data[stride + 3] = 255;
        max_val = Math.max(texture_data[stride], max_val);
      }
    }
    const texture = new THREE.DataTexture(texture_data, curr_size, curr_size);
    texture.needsUpdate = true;

    return {tex: texture, max_val: max_val};
  }
  const mountainGeometry = new THREE.PlaneGeometry(50000, 200, 200, 200);
  const {tex: mountainHeightMapTexture, max_val: max_val} = getMountainHeightMapTexture();

  const material = new THREE.MeshStandardMaterial({
    map: mountainHeightMapTexture,
    displacementMap: mountainHeightMapTexture,
    displacementScale: 230 / Math.min(max_val, 230) * 430,
    displacementBias: 0,
    color: 0xcac4bf
  });
  const mountains = new THREE.Mesh(mountainGeometry, material);
  mountains.rotation.x = -Math.PI / 2;
  mountains.castShadow = true;
  mountains.receiveShadow = true;
  mountains.material.needsUpdate = true;
  return mountains;
};

const WIDTH = 1280;
const HEIGHT = 960;
const ASPECT_RATIO = WIDTH / HEIGHT;
const NUM_GRASS = 100000;
const SUNLIGHT = {
  position: new THREE.Vector3(10, 200, -60),
  color: new THREE.Color(0.7, 0.7, 0.1)
}
const DESIRED_DT = 1.0 / 30.0;

const CLOCK = new THREE.Clock();
const PERLIN = new ImprovedNoise();

const RENDERER = new THREE.WebGLRenderer( { antialias: true } );
RENDERER.shadowMap.enabled = true;
RENDERER.shadowMap.autoUpdate = true;

const SCENE = new THREE.Scene();
SCENE.background = new THREE.Color().setRGB( 0.5, 0.5, 0.5 );

const CAMERA = new THREE.PerspectiveCamera(
  75,             // FOV (in degrees)
  ASPECT_RATIO,   // Aspect ratio
  0.1,            // Near clipping plane
  10000           // Far clipping plane
);
// Position the camera so it can view the ground plane and other objects
CAMERA.position.set(0, 110, 10);
// Look straight ahead
CAMERA.lookAt(0, 110, 0);

const FRUSTUM = new THREE.Frustum();
const PROJ_SCREEN_MATRIX = new THREE.Matrix4();

// Update frustum whenever the camera moves or changes
const updateFrustum = () => {
    CAMERA.updateMatrixWorld();
    PROJ_SCREEN_MATRIX.multiplyMatrices(CAMERA.projectionMatrix, CAMERA.matrixWorldInverse);
    FRUSTUM.setFromProjectionMatrix(PROJ_SCREEN_MATRIX);
}
updateFrustum();

const GPU_COMPUTE = makeGPUComputationRenderer3D(256, 256, 256, RENDERER);

window.onload = () => {
  CLOCK.start();

  const noiseTexture = GPU_COMPUTE.createTexture();
  const noiseVariable = GPU_COMPUTE.addVariable('textureNoise', getNoiseFragmentShader(), noiseTexture);
  GPU_COMPUTE.setVariableDependencies(noiseVariable, [noiseVariable]);
  const error = GPU_COMPUTE.init();
  if (error !== null) console.error(error);
  GPU_COMPUTE.compute();

  const cloud_noise_texture = GPU_COMPUTE.getCurrentRenderTarget(noiseVariable).texture;
  cloud_noise_texture.needsUpdate = false;

  const sky = makeSky();

  const plane = makePlane();

  const grass = makeGrass(plane, 180, 20, cloud_noise_texture);

  const clouds = makeClouds(cloud_noise_texture);
  clouds.position.set(0, 2050, -1000);
  clouds.scale.set(5000, 4000, 1000);

  const mountains = makeMountains();
  mountains.position.set(0, 0, -5550);

  SCENE.add(sky);
  SCENE.add(plane);
  SCENE.add(grass);
  SCENE.add(clouds);
  SCENE.add(mountains);

  SCENE.fog = new THREE.Fog(0x5973a9, 5450, 5550);

  let prev_update_time = 0;
  const animate = (time) => {
    if (time - prev_update_time > DESIRED_DT) {
      grass.material.uniforms.uTime.value += 0.001;
      grass.material.needsUpdate = true;
      cloud_noise_texture.needsUpdate = false;
      clouds.material.uniforms.frame.value += 0.001;

      composer.render(SCENE, CAMERA);

      prev_update_time = time;
    }
  }
  RENDERER.setPixelRatio(window.devicePixelRatio);
  RENDERER.setSize(WIDTH, HEIGHT);
  RENDERER.setAnimationLoop(animate);
  document.querySelector("body .wrapper section").appendChild(RENDERER.domElement);

  const renderPass = new RenderPass(SCENE, CAMERA);
  const kuwaharaPass = getKuwaharaPass(4, WIDTH, HEIGHT);

  const composer = new EffectComposer(RENDERER);
  composer.addPass(renderPass);
  // composer.addPass(bloomPass);
  composer.addPass(kuwaharaPass);

  const resizeObserver = new ResizeObserver((entries) => {
    if (entries.length !== 1) {
      console.error("More than one resize object detected!");
      return;
    }
    const section = entries[0];
    const new_display_width = section.contentRect.width;
    const new_display_height = new_display_width / ASPECT_RATIO;
    RENDERER.setSize(new_display_width, new_display_height);
  });
  resizeObserver.observe(document.querySelector("body .wrapper section"));
};