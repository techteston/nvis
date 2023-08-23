#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


st.set_page_config(page_title='Network Visualisation',page_icon=":tada:",layout="wide")

def get_lat_lon(fd_data,fd_mast,fv_data_field,fv_mast_field):
    fd_mast = fd_mast[[fv_mast_field,"Latitude","Longitude"]]
    df_lat_long = pd.merge(fd_data,fd_mast,left_on=fv_data_field,right_on=fv_mast_field,how="left")
    return(df_lat_long)

def create_nodes(fd_data,fv_value_name,fv_value,fv_size,fv_location,fv_color):
    
    fd_data = define_size_and_color(fd_data,fv_color,fv_size)
    
    nodes_trace = go.Scattermapbox(
        lat=fd_data['Latitude'],
        lon=fd_data['Longitude'],
        mode='markers',
        marker=dict(
            size=fd_data["node_size"] * 50,
            color=fd_data["node_colour"],
            opacity=0.7,
        ),
        text=fv_value_name+" at " + fd_data[fv_location] + ": " + fd_data[fv_value].astype(str),
    )
    return(nodes_trace)

def create_edges(fd_data,fv_src,fv_src_lat,fv_srv_lon,fv_des,fv_des_lat,fv_des_lon,fv_value,fv_value_name,fv_size,fv_color):

    fd_data = define_size_and_color(fd_data,fv_color,fv_size)  
    
    edges_trace = []
    for index, row in fd_data.iterrows():
        source = (row[fv_src_lat], row[fv_srv_lon])
        destination = (row[fv_des_lat], row[fv_des_lon])
        hover_text = f"Source: {row[fv_src]}\nDestination: {row[fv_des]}\n{fv_value_name}: {row[fv_value]}"
        edges_trace.append(go.Scattermapbox(
            lat=[source[0], destination[0]],
            lon=[source[1], destination[1]],
            mode='lines',
            line=dict(width=row["node_size"]*10, color=row["node_colour"]),  # Line width and color
            hoverinfo='text',  # Show custom hover text
            hovertext=hover_text,
        ))

    return(edges_trace)

def compute_actual_time(fd_transactions,date1,date2):
    fv = "Date Difference"
    # Convert the time differences to seconds and then round
    fd_transactions[fv] = (fd_transactions[date1] - fd_transactions[date2]).dt.total_seconds() / (60 * 60 * 24)

    # Calculate the mean and median of the date differences
    mean_difference = fd_transactions[fv].mean()
    median_difference = fd_transactions[fv].median()

    # Format mean and median to display with two decimal places
    mean_formatted = "{:.2f}".format(mean_difference)
    median_formatted = "{:.2f}".format(median_difference)
    mean_formatted = float(mean_formatted)
    median_formatted = float(median_formatted)
    return(mean_formatted,median_formatted)

def dimensions_weeks_supply(fv_dimension,fv_transactdata,fv_inven_data):
    df_dims = fv_transactdata[[fv_dimension]].drop_duplicates().reset_index(drop=True)
    fd_performance = pd.DataFrame()
    ctr = 0
    while ctr < len(df_dims): 
        va_var1 = df_dims.loc[ctr,fv_dimension]
        df_tran_subset = fv_transactdata[(fv_transactdata[fv_dimension]==va_var1)].copy()
        df_inv_subset = fv_inven_data[(fv_inven_data[fv_dimension]==va_var1)].copy()
        li_group_dims = [fv_dimension,"Product"]
        va_metric_label = "Units"
        va_inventory_label = "Inventory"
        va_date_label = "Order Date"
        df_wos = compute_weeks_supply(df_tran_subset,df_inv_subset,li_group_dims,va_metric_label,va_inventory_label,va_date_label)
        fd_performance = pd.concat([fd_performance, df_wos], ignore_index=True)
        ctr = ctr + 1
    
    return fd_performance

def compute_weeks_supply(fd_data,fd_inventory,fl_group_dims,fv_metric,fv_inventory,fv_date):
    fd_data2 = fd_data.groupby(fl_group_dims).sum(fv_metric).reset_index()
    
    # Calculate max and min dates
    max_date = fd_data[fv_date].max()
    min_date = fd_data[fv_date].min()

# Calculate the number of weeks between max and min dates
    weeks_difference = (max_date - min_date).days / 7
    
    fd_wos = pd.merge(fd_data2,fd_inventory,on=fl_group_dims,how="left")
    fd_wos["WoS"] = fd_wos[fv_inventory]/(fd_wos[fv_metric]/weeks_difference)
    return(fd_wos)

def define_size_and_color(fd_data,label_colour,label_size):

    # Normalize Node Size
    value_max = fd_data[label_size].max()
    fd_data["node_size"] =  fd_data[label_size]/value_max

    # Normalize Node Colour
    value_max = fd_data[label_colour].max()
    fd_data["node_color_norm"] =  fd_data[label_colour]/value_max

    norm = mcolors.Normalize(vmin=fd_data["node_color_norm"].min(), vmax=fd_data["node_color_norm"].max())
    colors = [(0, '#129678'), (0.3, '#E89C1D'), (1, '#DB1B39')]  # Example custom colors
    cmap = mcolors.LinearSegmentedColormap.from_list('CustomColormap', colors)
    def value_to_color(value):
        rgba = cmap(norm(value))
        hex_color = mcolors.rgb2hex(rgba)
        return hex_color

    # Apply the function to create the "COL" column with hex color codes
    fd_data["node_colour"] = fd_data["node_color_norm"].apply(value_to_color)
    
    return(fd_data)

with st.container():
    st.title('Network Visualisation')
    st.write("Visualizing the Entire Supply Chain")

file = st.file_uploader("Upload an Excel File",type=['xlsx'])
if file is not None:
    st.success('File Uploaded Successfully!', icon="✅")
#    st.balloons()
    try:
        df_products = pd.read_excel(file,sheet_name="ProductMaster")
        df_customers = pd.read_excel(file,sheet_name="CustomerMaster")
        df_dcs = pd.read_excel(file,sheet_name="DCMaster")
        df_factories = pd.read_excel(file,sheet_name="FactoryMaster")

        df_dccustomerinfo = pd.read_excel(file,sheet_name="DCCustomer")
        df_factorydcinfo = pd.read_excel(file,sheet_name="FactoryDC")

        df_customerinventory = pd.read_excel(file,sheet_name="CustomerInventory")
        df_dcinventory = pd.read_excel(file,sheet_name="DCInventory")

        li_fields = [""]
        li_dates = ["Order Date","Demand Date","Delivery Date"]
        df_salestransactions = pd.read_excel(file,sheet_name="SalesTransactions",parse_dates=li_dates)
        li_dates = ["Order Date","Demand Date","Delivery Date"]
        df_disttransactions = pd.read_excel(file,sheet_name="DistributionTransactions",parse_dates=li_dates)
        with st.expander("Uploaded Data", expanded=False):
            with st.container():
                cust_mast,cust_inv = st.columns((2,1))
                with cust_mast:
                    st.caption("Customers")
                    st.dataframe(data=df_customers, width=None, height=None,hide_index=1)
                with cust_inv:
                    st.caption("Customer Inventory")
                    st.dataframe(data=df_customerinventory, width=None, height=None,hide_index=1)
    except:
        st.error('Upload Failed')
    if len(df_customers) > 0:
        with st.container():
                prd_sel,cus_sel,dc_sel = st.columns((2,4,2))
                custselect = ""
                prodselect = ""
                dcselect = ""
                factselect = ""
                with prd_sel:
                    prodselect = st.multiselect('Select a Product',(df_products["Product"]),default="P1")
                with cus_sel:
                    custselect = st.multiselect('Select a Customer',(df_customers["Customer Full"]),default=df_customers["Customer Full"])
                with dc_sel:
                    dcselect = st.multiselect('Select a DC',(df_dcs["DC"]),default=df_dcs["DC"])
#                with fac_sel:
#                    factselect = st.multiselect('Select a Factory',(df_factories["Factory"]),default=df_factories["Factory"])
        with st.expander("Inventory View", expanded=False):
            if custselect == "" or prodselect=="":
                st.write('')         
            else:
                # Filter the selected data
                df_cust_spec = df_customerinventory.loc[(df_customerinventory["Customer"].isin(custselect))&(df_customerinventory["Product"].isin(prodselect))]
                df_dc_spec = df_dcinventory.loc[(df_dcinventory["DC"].isin(dcselect))&(df_dcinventory["Product"].isin(prodselect))]
                df_tran_spec = df_salestransactions.loc[(df_salestransactions["Product"].isin(prodselect))&
                                                        (df_salestransactions["Customer"].isin(custselect))&
                                                    (df_salestransactions["DC"].isin(dcselect))].copy()
                # Process for unique combinations
                df_dims = df_tran_spec[["Product","Customer","DC"]].drop_duplicates().reset_index(drop=True)
                # Find the Stated Information
                df_cust_dc_lt = pd.merge(df_dims,df_dccustomerinfo,left_on=["Customer","DC"],right_on=["Customer","DC"],how="left")
                # Compute Actual Delivery Performance
                ctr = 0
                while ctr < len(df_cust_dc_lt):
                    va_var1 = df_cust_dc_lt.loc[ctr,"Customer"]
                    va_var2 = df_cust_dc_lt.loc[ctr,"DC"]
                    df_subset = df_tran_spec[(df_tran_spec["Customer"]==va_var1)&(df_tran_spec["DC"]==va_var2)].copy()
                # Compute the Average and Median Delivery Times
                    avg,med = compute_actual_time(df_subset,"Delivery Date","Order Date")
                    df_cust_dc_lt.loc[ctr,"Avg Calc"] = avg
                    df_cust_dc_lt.loc[ctr,"Med Calc"] = med
                    ctr = ctr +1

                # Inventory Performance (Weeks of Supply)
                va_dimension = "DC" 
                df_dc_perf =  dimensions_weeks_supply(va_dimension,df_tran_spec,df_dcinventory)
                va_dimension = "Customer" 
                df_cust_perf =  dimensions_weeks_supply(va_dimension,df_tran_spec,df_customerinventory)

                # Compose Nodes
                df_cust_locations = get_lat_lon(fd_data=df_cust_perf,fd_mast=df_customers,fv_data_field="Customer",fv_mast_field="Customer Full")

                df_dc_locations = get_lat_lon(fd_data=df_dc_perf,fd_mast=df_dcs,fv_data_field="DC",fv_mast_field="DC")

                df_cust_locations.rename(columns={"Customer":"Location"},inplace=True)
                li_fields = ["Location","Product","Units","Revenue","Inventory","WoS","Latitude","Longitude"]
                df_cust_locations = df_cust_locations[li_fields]

                df_dc_locations.rename(columns={"DC":"Location"},inplace=True)
                li_fields = ["Location","Product","Units","Revenue","Inventory","WoS","Latitude","Longitude"]
                df_dc_locations = df_dc_locations[li_fields]

                df_locations = pd.concat([df_cust_locations,df_dc_locations],ignore_index=True)

                df_nodes = create_nodes(fd_data=df_locations,fv_value_name="Inventory",fv_value="Inventory",
                                            fv_size="Inventory",fv_location="Location",fv_color="WoS")

                layout = go.Layout(mapbox=dict(
                    center=dict(lat=df_locations['Latitude'].mean(), lon=df_locations['Longitude'].mean()),
                    zoom=3,
                    style='open-street-map',),showlegend=False,)
                # Create the figure
                fig = go.Figure(data=[df_nodes], layout=layout)
                fig.update_layout(height=600)
                st.plotly_chart(fig,height=600)
        with st.expander("Inventory and Transportation View", expanded=False):

            new_var = 'Month_Year' 
            old_var = "Order Date"
            
            df_tran_spec[new_var] = df_tran_spec[old_var].dt.strftime('%b-%Y')
            df_dates=df_tran_spec[[old_var,new_var]].copy()
            df_dates.drop_duplicates(subset=[old_var],inplace=True)
            df_dates.sort_values(by=[old_var],inplace=True)
            df_dates = df_dates.reset_index(drop=True)

            start_dt = df_dates.loc[df_dates[old_var].idxmin(), new_var]
            end_dt = df_dates.loc[df_dates[old_var].idxmax(), new_var]
            li_dates = pd.unique(df_dates[new_var])

            sel_start_dt,sel_end_dt = st.select_slider(
                'Select a range of color wavelength',
                options=li_dates,
                value=(start_dt, end_dt)
                )

            search_index = df_dates.index[df_dates[new_var] == sel_start_dt].tolist()
            from_dt = df_dates.loc[search_index[0], old_var]
            search_index = df_dates.index[df_dates[new_var] == sel_end_dt].tolist()
            to_dt = df_dates.loc[search_index[0], old_var]

            df_tran_spec2 = df_tran_spec[(df_tran_spec[old_var]>=from_dt)&(df_tran_spec[old_var]<=to_dt)]



            va_metric = "Units"
            li_fields = ["Product","Customer","DC"]

            df_dim_groups = df_tran_spec2.groupby(li_fields).sum(va_metric).reset_index()
            li_fields = li_fields+[va_metric]
            df_dim_groups = df_dim_groups[li_fields]

            # Find the Stated Information
            df_cust_dc_lt = pd.merge(df_dim_groups,df_dccustomerinfo,left_on=["Customer","DC"],right_on=["Customer","DC"],how="left")
            # Compute Actual Delivery Performance
            ctr = 0
            while ctr < len(df_cust_dc_lt):
                va_var1 = df_cust_dc_lt.loc[ctr,"Customer"]
                va_var2 = df_cust_dc_lt.loc[ctr,"DC"]
                df_subset = df_tran_spec2[(df_tran_spec2["Customer"]==va_var1)&(df_tran_spec2["DC"]==va_var2)].copy()
            # Compute the Average and Median Delivery Times
                avg,med = compute_actual_time(df_subset,"Delivery Date","Order Date")
                df_cust_dc_lt.loc[ctr,"Avg"] = avg
                df_cust_dc_lt.loc[ctr,"Med"] = med
                ctr = ctr +1
            df_cust_dc_lt["Del Performance"] = df_cust_dc_lt["Avg"]/df_cust_dc_lt["Days"]
            df_cust_dc_lt_locations = get_lat_lon(df_cust_dc_lt,df_customers,"Customer","Customer Full")
            df_cust_dc_lt_locations.rename(columns={"Latitude":"Source_Latitude","Longitude":"Source_Longitude"},inplace="True")
            df_cust_dc_lt_locations = get_lat_lon(df_cust_dc_lt_locations,df_dcs,"DC","DC")
            df_cust_dc_lt_locations.rename(columns={"Latitude":"Destination_Latitude","Longitude":"Destination_Longitude"},inplace="True")
            dt_edges = create_edges(df_cust_dc_lt_locations,"DC","Source_Latitude","Source_Longitude",
                                "Customer","Destination_Latitude","Destination_Longitude","Days","Days","Units","Del Performance")
            fig2 = go.Figure(data=[df_nodes,*dt_edges], layout=layout)
            fig2.update_layout(height=600)
            st.plotly_chart(fig2,height=600)
    else:
        st.write('')

else:
    st.write('')