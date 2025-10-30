from django.urls import path
from .views import index,pie_chart_api,total_data,recent_records_api,waveform_by_id,edit_data,upload_plot,get_pqrst_data
urlpatterns = [
    path('', index, name='report_index'),
    path("pie-chart/", pie_chart_api, name="pie_chart_api"),
    path("recent-records/", recent_records_api, name="recent_records_api"),
    # path("waveforms/",waveform,name="waveforms")
    path("waveform/<str:collection_name>/<str:record_id>/", waveform_by_id, name="waveform_by_id"),
    path('edit_data/',edit_data, name="edit_data"),
    path("upload_plot/", upload_plot, name="upload_plot"),    
    path('get_pqrst_data/', get_pqrst_data, name='get_pqrst_data'),
    path('total/', total_data, name='total_data'),

]