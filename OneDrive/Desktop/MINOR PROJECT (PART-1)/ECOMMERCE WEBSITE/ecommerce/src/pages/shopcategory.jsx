import React, { useContext } from 'react'
import { ShopContext } from '../context/shopcontext'
import dropdown_icon from'../components/images/Assets/dropdown_icon.png';
import Item from '../components/item/item';
import './CSS/shopcategory.css'
export default function Shopcategory(props) {
  const {all_product}=useContext(ShopContext)
  return (
    <div className='shop-category'>
      <img className='shopcategory-banner' src={props.banner}></img>
    <div className="shopcategory-indexsort">
      <p>
        <span>
          showing 1-12
        </span>out of 30 Products
      </p>
      <div className="shopcategory-sort">
        sort by <img src={dropdown_icon} alt="" />
      </div>
    </div>
    <div className="shopcategory-products">
    
    {all_product.map((item,i)=>{
      if (props.category===item.category) {
        return<Item key={i} id={item.id} name={item.name} image={item.image} new_price={item.new_price} old_price={item.old_price}/>
      }
      else{
        return null;
      }
    })}
    </div>
    <div className='shopcategory-loadmore'>
      Explore More
    </div>
    </div>
  )
}

