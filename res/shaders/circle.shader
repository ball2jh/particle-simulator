#shader vertex
#version 330 core

layout(location = 0) in vec4 vertex_position;
layout(location = 1) in float fade;

out float v_fade;

void main()
{
    gl_Position = vertex_position;
    v_fade = fade;
}

#shader fragment
#version 330 core

in float v_fade;

out vec4 fragColor;

void main()
{
    // Parameters
    vec3 circleColor = vec3(0.85, 0.35, 0.2);
    float thickness = 1.0;
    float fade = v_fade / 800;

    // -1 -> 1 local space, adjusted for aspect ratio
    vec2 uv = (gl_FragCoord.xy - vec2(800.0, 800.0) * 0.5) / min(800, 800) * 2.0;
    float aspect = 1.0;
    uv.x *= aspect;
    
    // // Calculate distance and fill circle with white
    float distance = 1.0 - length(uv);
    vec3 color = vec3(smoothstep(0.0, fade, distance));
    color *= vec3(smoothstep(thickness + fade, thickness, distance));

    // Set output color
    fragColor = vec4(color, 1.0);
    fragColor.rgb *= circleColor;
}
