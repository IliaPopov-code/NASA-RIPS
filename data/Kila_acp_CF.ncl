;*************************************************


;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"   
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;************************************************

begin
 
 

  label_phase =  "CFLO_onebar"
  wks_type = "eps"  
  wks_type@wkPaperSize = "A4"
  wks   = gsn_open_wks (wks_type, "Kila_plots/"+label_phase)                  ; open workstation
   
   
  colormp =  "BlueWhiteOrangeRed"
  gsn_define_colormap(wks, colormp)   
  plot = new(8,graphic)                                    ; create a plot array
  plot_zmean = new(2,graphic)
   
  fontsz = 0.04
  
;************************************************
; settings
;************************************************
  res                      = True                   ; plot mods desired   
  res@cnLevelSelectionMode = "ManualLevels"         ; manually select levels
  
  res@cnFillOn             = True                   ; turn on color
  res@cnLinesOn            = False                  ; no contour lines
  res@cnLineLabelsOn       = False                  ; no contour lines
  res@gsnSpreadColors      = True                   ; use full range of colors
  res@lbLabelAutoStride    = True                  ; optimal labels  
 
  res@gsnDraw              = False                   ; don't draw yet
  res@gsnFrame             = False                   ; don't advance frame yet
  res@tiMainFontHeightF    = 16
  res@tiMainFont           = "Helvetica-Bold"
  res@lbLabelBarOn          = False
  ;res@tmXBLabelsOn         = False              ; do not draw bottom labels
  ;res@tmXBOn               = False              ; no bottom tickmarks
  res@tmXBLabelFontHeightF = fontsz                 ; resize tick labels
  res@tmYLLabelFontHeightF = fontsz  
  res@gsnStringFontHeightF  =fontsz
  res@mpCenterLonF          = -165
  res@cnInfoLabelOn = False                     ; turn off cn info label
  res@gsnMajorLatSpacing = 5
  res@gsnMajorLonSpacing = 10 
  res@lbLabelFontHeightF =fontsz
  res@pmLabelBarOrthogonalPosF = .3
  res@lbBoxEndCapStyle = "TriangleBothEnds"

  res@tmXBLabelFontHeightF = fontsz                 ; resize tick labels
  res@tmYLLabelFontHeightF = fontsz   
  res@tmYRLabelFontHeightF = fontsz    
  res@gsnStringFontHeightF  =fontsz
 
  res@mpGridAndLimbOn        = True
  
  res@mpGridLineDashPattern  = 5                  ; lat/lon lines dashed
  res@mpGridLatSpacingF      = 5.0
  res@mpGridLonSpacingF      = 5.0
  ;res@mpGridAndLimbDrawOrder = "PreDraw"          ; Draw grid first
  res@mpGridLineThicknessF =  0.25
   
     res@gsnLeftString      =  ""          
     
     ; donif: doing it this way we can emphasize some things
     
     res@cnLevelSelectionMode = "ExplicitLevels"         ; manually select levels
     ;res@cnLevels = (/-0.05, 0.0, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.11, 0.12, 0.13, 0.14, 0.18, 0.22/)
     res@cnLevels = (/-0.1, -0.04, 0.0, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.17, 0.19, 0.25/)
     res@cnLineLabelInterval = 1
   

  res@sfXCStartV = -180   ; Define where contour plot
  res@sfXCEndV   = -155   ; should lie on the map plot.
  res@sfYCStartV = 10
  res@sfYCEndV   = 25
  
  res@mpLimitMode = "LatLon"    ; Limit the map view.
  res@mpMinLonF   = -180
  res@mpMaxLonF   = -155
  res@mpMinLatF   = 10
  res@mpMaxLatF   = 25
  res@mpPerimOn   = True        ; Turn on map perimeter.
  ; for some reason in the original plots, the x labels were (180, 170W, 160W) instead of (180W, 170W, 160W)
  res@tmXBMode        = "Explicit"								; specify x axis ticks/labels
  res@tmXBValues      = (/-180,-170,-160/)
  res@tmXBLabels      = (/"180W","170W","160W"/)
  res@gsnAddCyclic = False  ; don't wrap data around
   
; read in and plot
;************************************************ 
  data2008 = addfile("/discover/nobackup/khbreen/python_scripts/KILA_NC/JJA2008.nc4","r") 
  data2018 = addfile("/discover/nobackup/khbreen/python_scripts/KILA_NC/MJJ2018.nc4","r") 
  
;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
;                                        CF
;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ;========= build the maps first
  ; 2008 
  
  v10x2008 = data2008->CFLO_v10x(:, :)
  v1clim2008 = data2008->CFLO_v1clim(:, :)
  vmodis2008 = data2008->CFLO_vmodis(:,:)
  
  ;minN =  0
  ;maxN =  0.25
     
  ;   res@cnMinLevelValF       = minN             
  ;   res@cnMaxLevelValF       = maxN           
  ;   res@cnLevelSpacingF      = (maxN-minN)/30.0
	 
  res@tmXBOn = False  ; turn off x labels - turn back on for bottom row and zonal plots 
  res@gsnLeftString      =  "2008" 
  res@gsnCenterString      =  "MODIS anomaly"
  plot(0) = gsn_csm_contour_map(wks,vmodis2008, res)  ; MODIS JJA 2008
  res@tmYLOn = False  ; turn off y labels 
  res@gsnLeftString      =  "" 
  res@gsnCenterString      =  "GEOS 1x - GEOS~B~clim"
  plot(1) = gsn_csm_contour_map(wks,v1clim2008, res)  ; 1x-clim JJA 2008
  res@gsnCenterString      =  "GEOS 1x - 0x"
  plot(2) = gsn_csm_contour_map(wks,v10x2008, res)  ; 1x-0x JJA 2008
  
   ; 2018

  v10x2018 = data2018->CFLO_v10x(:, :) 
  v1clim2018 = data2018->CFLO_v1clim(:, :)
  vmodis2018 = data2018->CFLO_vmodis(:,:)

  res@tmXBOn = True
  res@tmYLOn = True
  res@gsnCenterString      =  "" 
  res@gsnLeftString      =  "2018" 
  plot(4) = gsn_csm_contour_map(wks,vmodis2018, res)  ; MODIS MJJ 2018
  res@tmYLOn = False 
  res@gsnLeftString      =  "" 
  plot(5) = gsn_csm_contour_map(wks,v1clim2018, res)  ; 1x-clim MJJ 2018
  plot(6) = gsn_csm_contour_map(wks,v10x2018, res)  ; 1x-0x MJJ 2018
 
  
   ;====== now zonal mean plots. =======
	 
  resxy = res 
  resxy@xyLineColors      = (/"black","green"/)  ; line colors geos_1xm0x, geos_1xmclim
  resxy@xyLineThicknesses = (/  2.0,   2.0/)  
  
  zonal_siz = 0.42 ;DONIF these have to be set by trial an error to align the zonal means and contours  
  resxy@vpWidthF             = zonal_siz                  ; change aspect ratio of plot so it fits better
  resxy@vpHeightF            = zonal_siz  ; 
 
  resxy@trYMaxF  =  25.0
  resxy@trYMinF =  10.0
  resxy@gsLineThickness = 3
  resxy@tmXMajorGrid                = True          ; implement x grid 
  resxy@tmXMajorGridThicknessF      = 0.25           ; 2.0 is default
  resxy@tmXMajorGridLineDashPattern = 2             ; select short dash lines
  
  resxy@tmYMajorGrid                = True          ; implement y grid
  resxy@tmYMajorGridThicknessF      = 0.25           ; 2.0 is default
  resxy@tmYMajorGridLineDashPattern = 2             ; select short dash lines
  
  resxy@tmXBPrecision = 1 ; only one decimal
  resxy@tmXBMode        = "Automatic"
  resxy@tmXBMaxTicks = 4
  resxy@tmYUseLeft = False
  resxy@tmYLOn = False ; turn off L y axis labels 
  ;resxy@tmYRBorderOn = True
  resxy@tmYRLabelsOn = True
  resxy@tmYROn = True ; turn on R y axis labels 
  resxy@tmYRMode        = "Explicit"								; specify y axis ticks/labels
  resxy@tmYRValues      = (/10,15,20,25/)
  resxy@tmYRLabels      = (/"10N","15N","20N","25N"/)
  
  
  ;==2008=====
   
  printVarSummary(v10x2008)
  x = data2008->geoslat(:)
  zonal_all= new((/2, dimsizes(v10x2008&geoslat)/) ,typeof(v10x2008), v10x2008@_FillValue)
  xs= new((/2, dimsizes(v10x2008&geoslat)/) ,typeof(v10x2008), v10x2008@_FillValue)
  xs(0, :)  =  x ; unfortunately I don't know a better way to do this 
  xs(1, :)  =  x
  zonal_all(0,:) = data2008->CFLO_v10x_zonal(:)
  zonal_all(1,:) = data2008->CFLO_v1clim_zonal(:)
   
  resxy@trXMaxF = -.5
  resxy@trXMinF =  0.4
  plot(3) = gsn_csm_xy(wks, zonal_all,xs, resxy)          ; Zonal means 2008
  
  ; do modis zonal means  
  
  resmod =  resxy 
  resmod@gsnCenterString      =  "Zonal mean"
  
  resmod@xyLineColors      = (/"blue","red"/)  ; list must be longer than 1, the second string is a dummy

  modis_mean = data2008->CFLO_vmodis_zonal(:)
  plt_dum =  gsn_csm_xy(wks, modis_mean, modis_mean&modislat, resmod)  
  overlay(plot(3), plt_dum)
  resmod@gsnCenterString      =  ""
  
  ;====2018
 
  zonal_all(0,:) = data2018->CFLO_v10x_zonal(:)
  zonal_all(1,:) = data2018->CFLO_v1clim_zonal(:)
 
  
  resxy@trXMaxF = -.05
  resxy@trXMinF =  0.35
  plot(7) = gsn_csm_xy(wks,zonal_all, xs, resxy)          ; Zonal means 2018
  modis_mean = data2018->CFLO_vmodis_zonal(:)
  plt_dum1 =  gsn_csm_xy(wks, modis_mean, modis_mean&modislat, resmod)  
  overlay(plot(7), plt_dum1)


 
;***********************************************
; create combined plot
;************************************************


 resP            = True                         ; modify the panel plot
 resP@lbLabelAutoStride    = True
 resP@txString   =  ""; label_phase
  resP@lbTitleOn = True  ; turn on a title for the colorbar
  resP@lbTitleString = "CF" 
  resP@lbTitleOffsetF = 0.3
 resP@gsnFrame         = False               ; save panel until both ready
 resP@gsnPanelScalePlotIndex = 2 
 resP@gsnPanelLabelBar    = True 
 resP@lbLabelFontHeightF  = 0.018
 resP@gsnMaximize      = False                       ; use full page
 resP@gsnPanelXWhiteSpacePercent = 5  ; set x white space between panel plots
 resP@gsnPanelYWhiteSpacePercent = 5  ; set y white space between panel plots
 resP@lbTitleFontHeightF    = 0.016
 resP@lbTitlePosition = "Bottom"
 resP@pmLabelBarOrthogonalPosF = -.05
 resP@lbBoxEndCapStyle = "TriangleBothEnds"
 
  resP@lbBoxMinorExtentF    = 0.5
  resP@pmLegendParallelPosF= 0.3
  resP@lbLabelJust          = "CenterRight"
  resP@lbLabelOffsetF       = 0.5
  resP@vpXF               = 0.1
  resP@pmLabelBarWidthF      = 0.6
  resP@gsnPanelLeft = 0.1  ; set the left margin
  ;resP@gsnPanelRight = 0.1  ; set the right margin
  gsn_panel(wks,plot,(/2, 4/),resP)             ; now draw as one plot


  
end 
  
  
  
  
  
  
  
  
  
  
