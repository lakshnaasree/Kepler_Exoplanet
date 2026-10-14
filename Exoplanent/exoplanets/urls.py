from django.urls import path
from exoplanets import views


urlpatterns = [
    path(
        "",
        views.home,
        name="home",
    ),

    path(
        "explorer/",
        views.explorer,
        name="explorer",
    ),

    path(
        "visualizations/",
        views.visualizations,
        name="visualizations",
    ),

    path(
        "comparison/",
        views.comparison,
        name="comparison",
    ),
    path("kepler-summary/",
        views.kepler_summary,
        name="kepler_summary"),
]