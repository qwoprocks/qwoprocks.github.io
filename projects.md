---
title: Projects
---

{% capture mangaLogo %}
{{ "/assets/img/manga_logo.png" | relative_url }}
{% endcapture %}

{% capture includeGuts %}
{% include card.html link="https://seahhorse.github.io/projects/improving_image_representations_of_words_through_region-specific_loss_minimisation/"
                     linktarget="_blank"
                     image=mangaLogo
                     alt="A sample manga panel from the Manga109 dataset"
                     title="Improving Image Representations of Words through Region-specific Loss Minimisation"
                     description="How can we make text in generated images seem more real?" 
                     time="22 May 2024"
                     authors="Ming Chong Lim, Shao Xuan Seah"   %}
{% endcapture %}
{{ includeGuts | replace: '  ', ''}}


{% capture imgsynProj5Logo %}
{{ "/assets/img/imgsyn_proj5_logo.png" | relative_url }}
{% endcapture %}

{% capture includeGuts %}
{% include card.html link="https://www.andrew.cmu.edu/course/16-726-sp24/projects/mingchol/proj5/"
                     linktarget="_blank"
                     image=imgsynProj5Logo
                     alt="Logo for sketch2image"
                     title="Sketch2Image"
                     description="A project done for 16-726 Image Synthesis at CMU involving image synthesis through latent variable manipulation" 
                     time="Apr 2024"
                     authors="Ming Chong Lim"   %}
{% endcapture %}
{{ includeGuts | replace: '  ', ''}}


{% capture imgsynProj4Logo %}
{{ "/assets/img/imgsyn_proj4_logo.png" | relative_url }}
{% endcapture %}

{% capture includeGuts %}
{% include card.html link="https://www.andrew.cmu.edu/course/16-726-sp24/projects/mingchol/proj4/"
                     linktarget="_blank"
                     image=imgsynProj4Logo
                     alt="Logo for neural style transfer"
                     title="Neural Style Transfer"
                     description="A project done for 16-726 Image Synthesis at CMU involving texture synthesis and style transfer"
                     time="Mar 2024"
                     authors="Ming Chong Lim"   %}
{% endcapture %}
{{ includeGuts | replace: '  ', ''}}


{% capture routemakerLogo %}
{{ "/assets/img/routemaker_logo.jpg" | relative_url }}
{% endcapture %}

{% capture includeGuts %}
{% include card.html link="https://github.com/nandium/RouteMaker"
                     linktarget="_blank"
                     image=routemakerLogo
                     alt="Logo for RouteMaker"
                     title="RouteMaker (Front-end Developer)"
                     description="A cross-platform application utilizing machine learning to allow users to quickly create climbing routes and share them with the community" 
                     time="Apr 2021 - Sep 2023"
                     authors="Ming Chong Lim, Yar Khine Phyo"   %}
{% endcapture %}
{{ includeGuts | replace: '  ', ''}}


{% capture tbmLogo %}
{{ "/assets/img/tbm_logo.png" | relative_url }}
{% endcapture %}

{% capture includeGuts %}
{% include card.html link="https://github.com/AY2021S1-CS2103T-F11-4/tp"
                     linktarget="_blank"
                     image=tbmLogo
                     alt="Logo for Travelling BusinessMan"
                     title="Travelling BusinessMan (Software Developer)"
                     description="A brownfield project done in Java under the module CS2103T Software Engineering" 
                     time="Aug 2020 - Nov 2020"
                     authors="Ming Chong Lim"   %}
{% endcapture %}
{{ includeGuts | replace: '  ', ''}}
