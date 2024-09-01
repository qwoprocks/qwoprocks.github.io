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
