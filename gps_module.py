from geopy.geocoders import Nominatim
import requests

def get_gps_coordinates(location):
    geolocator = Nominatim(user_agent="blind_navigation")
    loc = geolocator.geocode(location)
    return (loc.latitude, loc.longitude) if loc else None

def get_route(start_coords, end_coords):
    api_url = f"https://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
    response = requests.get(api_url)
    route = response.json()
    return route["routes"][0]["geometry"] if "routes" in route else None
