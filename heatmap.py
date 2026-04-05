import pandas as pd
import folium

# join on business_id to create result df
gem_df = pd.read_csv("ca_restaurants_with_hidden_gem_flag.csv")
restaurants_df = pd.read_csv("ca_restaurants.csv")
df = pd.merge(gem_df , restaurants_df, on='business_id')

map_center = [df["latitude"].mean(), df["longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=6)

# add hidden gem restaurants to the map
for _, row in df.iterrows():
    color = "green" if row["is_hidden_gem"] else "red"
    
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.7
    ).add_to(m)
m.save("restaurant_map.html")

print("Map saved as restaurant_map.html")